# Cfg vs hyperparam

**Created**: 2023-02-11
**Closed**: 2023-02-12
**Priority**: 2
**Labels**: feature

## Description

In the master branch we only support cli args for VRE. In the dev branch we added support for `OmegaConf`, but now we removed cli support.

Ideally it should be both. The proposal is the following:
based on `--cfg_path`:
  - if set, will read vre hparams from cfg
  - if not set, the will read from CLI
  - if set and we also provide cli args, it will throw an error, so we have to pick either cfg or cli
