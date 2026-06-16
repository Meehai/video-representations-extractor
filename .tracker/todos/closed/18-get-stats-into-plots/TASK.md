# get_stats into plots

**Created**: 2023-11-01
**Closed**: 2024-05-15
**Priority**: 2
**Labels**: docs/examples/user api, feature

## Description

2 plots at the end of each vre run:
- x axis = frame, y axis = each repres -> should show us if some frames are slower than others but also provide a Nice time plot
- histogram with average per representation for all times -> should give us average +/- std as well; perhaps put a log() scale too. Helps with knowing what to optimize
  -> we could generate this at end to end test and upload to imgur as well
