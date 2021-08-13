import numpy as np
from matplotlib.cm import hot
from typing import List

from .utils import scale_depth
from ..representation import Representation


class DepthEnsemble(Representation):
    def __init__(self, name, dependencies:List[Representation], saveResults:str, dependencyAliases:List[str],
                 voteType:str="mean", mode:str="all", reference:str=None,
                 interpolateMissingScaleRef:str="none", neighborhood:int=300, neighborhoodStep:int=10,
                 minScalingInlierRatio:float=0.0,
                 depthScalingMin:float=50, depthScalingMax:float=150,
                 depthScalingThres:float=5, depthScalingIter:int=10,
                 minDepth:float=None, maxDepth:float=None):
        super().__init__(name, dependencies, saveResults, dependencyAliases)
        assert voteType in ("mean", "median"), "Invalid voting type %s" % voteType
        assert mode in ("all", "fill_missing", "all_but_ref"), "Invalid ensemble mode %s" % mode
        assert mode not in ("fill_missing", "all_but_ref") or reference is not None, \
            "Set a reference when mode is not 'all'"
        assert interpolateMissingScaleRef in ("none", "nearest", "linear", "median_neighborhood"), \
            "Invalid scale interpolation %s" % interpolateMissingScaleRef
        assert 0 <= minScalingInlierRatio <= 1.0, "minScalingInlierRatio %.2f must be in [0, 1]" % minScalingInlierRatio
        assert neighborhood > 0, \
            "Neighborhood for scale calculation must be a positive number of frames"
        assert neighborhoodStep > 0, \
            "Neighborhood step for scale calculation must be a positive number of frames"
        assert neighborhood % neighborhoodStep == 0, "Neighborhood must be multiple of neighborhood step"
        self.voteFn = {"mean": np.nanmean, "median": np.nanmedian}[voteType]
        self.mode = mode
        self.reference = reference
        self.interpolateMissingScaleRef = interpolateMissingScaleRef
        self.minScalingInlierRatio = minScalingInlierRatio
        self.neighborhood = neighborhood
        self.neighborhoodStep = neighborhoodStep
        self.depthScalingRange = (depthScalingMin, depthScalingMax)
        self.depthScalingThres = depthScalingThres
        self.depthScalingIter = depthScalingIter
        self.depthRange = (minDepth, maxDepth)
        assert all(self.depthRange) or not any(self.depthRange), "Specify both minDepth and maxDepth"
        assert len(dependencies) >= 2, "Expected at least two depth methods!"
        self.depths = dependencies
        assert any([d.name == reference for d in self.depths]), \
            "Reference method %s not found in dependencies" % reference
        self.scalingFactors = None

    def make(self, t):
        repr_to_depth = {}
        ref_repr = None
        any_scaled = False
        for depth_repr in self.depths:
            result_depth = depth_repr[t]
            depth = result_depth["data"].copy()
            # put nan where invalid
            if "rangeValid" in result_depth["extra"]:
                depth = depth
                rangeValid = result_depth["extra"]["rangeValid"]
                depth[np.logical_or(depth <= rangeValid[0], depth >= rangeValid[1])] = np.nan
            # scale depth to given range
            if "rangeScaled" in result_depth["extra"]:
                rangeScaled = result_depth["extra"]["rangeScaled"]
                depth = depth * (rangeScaled[1] - rangeScaled[0]) + rangeScaled[0]
                any_scaled = True
            if self.reference is not None and depth_repr.name == self.reference:
                ref_repr = depth_repr
            repr_to_depth[depth_repr] = depth
        ratioInfo = {}
        if ref_repr is not None:
            # scale other depths to the reference one (likely in meters)
            repr_to_depth_scaled = {}
            for repr, depth in repr_to_depth.items():
                if repr == ref_repr:
                    repr_to_depth_scaled[repr] = depth
                    ratioInfo[repr.name] = {"value": 1, "isInterpolated": False}
                    self.set_scaling_factor(t, repr, 1)
                    continue
                scaled_depth, scale, mask = scale_depth(repr_to_depth[ref_repr], depth, self.depthScalingRange,
                                          self.depthScalingThres, self.depthScalingIter)
                inlierRatio = np.count_nonzero(mask) / mask.size
                if inlierRatio >= self.minScalingInlierRatio and scale >= 0:
                    repr_to_depth_scaled[repr] = scaled_depth
                    ratioInfo[repr.name] = {"value": scale, "isInterpolated": False, "inlierRatio": inlierRatio}
                    self.set_scaling_factor(t, ref_repr, scale)
                elif self.interpolateMissingScaleRef != "none":
                    self.set_scaling_factor(t, repr, -1)
                    s, interp_info = self.get_interpolated_scaling_factor(t, repr)
                    repr_to_depth_scaled[repr] = s * depth
                    ratioInfo[repr.name] = {"value": s, "isInterpolated": True, "inlierRatio": inlierRatio,
                                            "interp_info": interp_info, }
                else:
                    self.set_scaling_factor(t, repr, -1)
                    ratioInfo[repr.name] = {"value": -1, "isInterpolated": False, "inlierRatio": inlierRatio}
            repr_to_depth = repr_to_depth_scaled
        extra = {"ratioInfo": ratioInfo}
        depths = list([depth for r, depth in repr_to_depth.items() if self.mode != "all_but_ref" or r != ref_repr])
        if len(depths) > 0:
            agg_depth = self.voteFn(np.stack(depths), axis=0) if len(depths) > 1 else depths[0]
        else:
            agg_depth = np.full_like(next(iter(repr_to_depth.values())), np.nan)
        if self.mode in ("all", "all_but_ref"):
            depth = agg_depth
        elif self.mode == "fill_missing":
            ref_depth = repr_to_depth[ref_repr].copy()
            missing = ~np.isfinite(ref_depth)
            ref_depth[missing] = agg_depth[missing]
            depth = ref_depth
        if all(self.depthRange):
            depth = np.clip(depth, *self.depthRange)
            extra["rangeValid"] = (0, 1)
        if any_scaled:
            if np.any(~np.isnan(depth)):
                depthRange = (np.nanmin(depth), np.nanmax(depth))
                depth = (depth - depthRange[0]) / (depthRange[1] - depthRange[0])
                extra["rangeScaled"] = depthRange
            depth[~np.isfinite(depth)] = 1
        return {"data": depth, "extra": extra}

    def makeImage(self, x):
        Where = np.where(x["data"] == 1)
        y = x["data"]
        assert y.min() >= 0 and y.max() <= 1
        y = hot(y)[..., 0:3]
        y = np.uint8(y * 255)
        y[Where] = [0, 0, 0]
        return y

    def setup(self):
        if self.interpolateMissingScaleRef != "none" and self.scalingFactors is None:
            # nan are not yet computed scaling factors while -1 are invalid scaling factors
            self.scalingFactors = {repr.name: np.full(len(self.video), np.nan) for repr in self.depths}

    def get_scaling_factor(self, t, repr):
        s = self.scalingFactors[repr.name][t]
        if np.isfinite(s):
            return s

        data = self[t]
        for a_repr, ratio_info in data["extra"]["ratioInfo"].items():
            if not ratio_info["isInterpolated"]:
                # keep only scaling factors obtained directly from aligning depths
                self.scalingFactors[a_repr][t] = ratio_info["value"]
            else:
                self.scalingFactors[a_repr][t] = -1
        return self.scalingFactors[repr.name][t]

    def set_scaling_factor(self, t, repr, s):
        if self.scalingFactors is not None:
            self.scalingFactors[repr.name][t] = s

    def get_interpolated_scaling_factor(self, t, repr):
        N = len(self.video)
        # advanced left and right at the same time (at most t frames to the left and N - 1 - t to the right)
        # and load/compute scales
        max_dt = max(t, N - 1 - t)
        for dt in range(self.neighborhoodStep, 1 + max_dt, self.neighborhoodStep):
            left_t = t - dt
            if left_t >= 0:
                self.get_scaling_factor(left_t, repr)
            right_t = t + dt
            if right_t < len(self.video):
                self.get_scaling_factor(right_t, repr)
            # try to interpolate with current loaded scales
            can_interpolate, s, extra = self.try_scale_interpolation(t, repr)
            if can_interpolate:
                return s, extra
            # otherwise keep loading scale factors for neighborhood frames

    def try_scale_interpolation(self, t, repr):
        clip_s = self.scalingFactors[repr.name]
        if self.interpolateMissingScaleRef in ("nearest", "linear"):
            closest_left_t = None
            for t_left in range(t - self.neighborhoodStep, 0, -self.neighborhoodStep):
                if not np.isfinite(clip_s[t_left]):
                    return False, None, None
                if clip_s[t_left] >= 0:
                    closest_left_t = t_left
                    break
            closest_right_t = None
            for t_right in range(t + self.neighborhoodStep, len(clip_s), +self.neighborhoodStep):
                if not np.isfinite(clip_s[t_right]):
                    return False, None, None
                if clip_s[t_right] >= 0:
                    closest_right_t = t_right
                    break

            # Did not find even a single valid scaling factor
            assert any((closest_left_t, closest_right_t)), "Depth for %s can not be aligned at all" % repr.name

            # return one of the scaling factors if the other one is missing
            closest_ts = (closest_left_t, closest_right_t)
            if closest_left_t is None:
                s_right = clip_s[closest_right_t]
                return True, s_right, closest_ts
            if closest_right_t is None:
                s_left = clip_s[closest_left_t]
                return True, s_left, closest_ts

            s_left = clip_s[closest_right_t]
            s_right = clip_s[closest_left_t]
            # interpolated the two scaling factors if both are present
            d_left = t - closest_left_t
            d_right = closest_right_t - t
            if self.interpolateMissingScaleRef == "nearest":
                if d_left < d_right:
                    return True, s_left, closest_ts
                elif d_left > d_right:
                    return True, s_right, closest_ts
                else:
                    return True, (s_left + s_right) / 2, closest_ts
            elif self.interpolateMissingScaleRef == "linear":
                s = (s_left * d_left + s_right * d_right) / (d_right + d_left)
                return True, s, closest_ts
        elif self.interpolateMissingScaleRef in ("median_neighborhood"):
            # TODO - problem with maximum recursion exceeded
            # if self.interpolateMissingScaleRef == "median_all":
            #     neighborhood_s = clip_s
            #     neighborhood = (0, len(clip_s) - 1)
            if self.interpolateMissingScaleRef == "median_neighborhood":
                neighborhood_start = max(0, t - self.neighborhood)
                neighborhood_end = min(len(clip_s), t + self.neighborhood)
                neighborhood_s = clip_s[neighborhood_start:neighborhood_end + 1:self.neighborhoodStep]
                neighborhood = (neighborhood_start, neighborhood_end - 1)
            if not np.all(np.isfinite(neighborhood_s)):
                return False, None, None
            valid_neighborhood_s = neighborhood_s[neighborhood_s > 0]
            assert len(valid_neighborhood_s) > 0, "Neighborhood too small for median scale calculation for frame %d" % t
            s = np.median(valid_neighborhood_s)
            return True, s, neighborhood

