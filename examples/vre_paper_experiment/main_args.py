image_format=["not-set", "jpg"]
resolutions=[[240,320], [540,960], [720,1280], [1080,1920]]
batch_sizes=[1,5,20]
num_frames=[100, 200, 300, 400, 500]
devices=["cuda", "cpu"]
# args = [(i, os, 1, nf) for i in image_format for os in resolutions for nf in num_frames]

# RGB + HSV experiment
if False:
  args = []
  # I, OS, BS, NF
  args.extend([(False, "not-set", resolution, 1, 100) for resolution in resolutions])
  args.extend([(False, "jpg", resolution, 1, 100) for resolution in resolutions])
  args.extend([(True, "jpg", resolution, 1, 100) for resolution in resolutions])

  for c,i,os,bs,nf in args:
    print(f"COMPRESS={c} IMAGE_FORMAT={i} OUTPUT_SIZE='{os}' BATCH_SIZE={bs} vre video_{os[0]}_{os[1]}.mp4 -o data_c{c}_n{nf}_{bs}_{i}_{os[0]}_{os[1]}/ --config_path cfg_rgb.yaml --frames 0..{nf} --output_dir_exists_mode skip_computed")

args = []
# I, OS, BS, NF, DVC
args.extend([(False, "not-set", os, bs, 100, dvc) for dvc in devices for bs in batch_sizes for os in resolutions])

#phg-mae
for c,i,os,bs,nf,dvc in args:
  print(f"VRE_DEVICE={dvc} COMPRESS={c} IMAGE_FORMAT={i} OUTPUT_SIZE='{os}' BATCH_SIZE={bs} vre video_{os[0]}_{os[1]}.mp4 -o safeuav_{dvc}_c{c}_n{nf}_{bs}_{i}_{os[0]}_{os[1]}/ --config_path cfg_rgb_safeuav.yaml --frames 0..{nf} --output_dir_exists_mode skip_computed")

#dpt
# for c,i,os,bs,nf,dvc in args:
#   print(f"VRE_DEVICE={dvc} COMPRESS={c} IMAGE_FORMAT={i} OUTPUT_SIZE='{os}' BATCH_SIZE={bs} vre video_{os[0]}_{os[1]}.mp4 -o dpt_{dvc}_c{c}_n{nf}_{bs}_{i}_{os[0]}_{os[1]}/ --config_path cfg_rgb_dpt.yaml --frames 0..{nf} --output_dir_exists_mode skip_computed")
