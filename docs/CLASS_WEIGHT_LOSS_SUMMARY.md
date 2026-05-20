# Class Weight Loss Summary

The table uses 3-seed ensemble results. Duplicate `ntl_boost3` rows from the initial and search2 runs are identical; the later `search2` row is retained.

| weight_mode | val_accuracy | val_macro_f1 | val_macro_auroc | test_accuracy | test_macro_f1 | test_macro_auroc | test_HGC_f1 | test_LGC_f1 | test_NTL_f1 | test_NST_f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0.8149 | 0.6892 | 0.9555 | 0.5767 | 0.4987 | 0.8084 | 0.5731 | 0.4630 | 0.0000 | 0.9589 |
| sqrt_inverse | 0.7110 | 0.6228 | 0.9315 | 0.5873 | 0.5370 | 0.8317 | 0.5036 | 0.5714 | 0.1379 | 0.9351 |
| pow0.75 | 0.8149 | 0.8212 | 0.9500 | 0.4603 | 0.4776 | 0.7834 | 0.3284 | 0.4179 | 0.2500 | 0.9143 |
| inverse | 0.7240 | 0.7564 | 0.9014 | 0.6032 | 0.5786 | 0.8158 | 0.6036 | 0.4792 | 0.3000 | 0.9315 |
| ntl_boost2 | 0.8539 | 0.8247 | 0.9715 | 0.5344 | 0.5451 | 0.8064 | 0.4196 | 0.4921 | 0.3243 | 0.9444 |
| ntl_boost2.5 | 0.8571 | 0.8463 | 0.9688 | 0.5714 | 0.5903 | 0.8090 | 0.5170 | 0.5088 | 0.4400 | 0.8955 |
| ntl_boost3 | 0.8279 | 0.8193 | 0.9645 | 0.5556 | 0.5821 | 0.8146 | 0.4857 | 0.4957 | 0.4815 | 0.8657 |
| ntl_boost3.5 | 0.8019 | 0.7935 | 0.9607 | 0.5238 | 0.5570 | 0.8132 | 0.4154 | 0.4961 | 0.4727 | 0.8438 |
| ntl_boost4 | 0.7695 | 0.7505 | 0.9506 | 0.5873 | 0.6075 | 0.8277 | 0.5077 | 0.5691 | 0.4746 | 0.8788 |
| ntl_boost5 | 0.7695 | 0.7575 | 0.9284 | 0.5608 | 0.5729 | 0.7966 | 0.4202 | 0.5833 | 0.3692 | 0.9189 |
| custom_1_1_2.5_1.1 | 0.8052 | 0.8051 | 0.9541 | 0.5714 | 0.6003 | 0.8162 | 0.4672 | 0.5000 | 0.4898 | 0.9444 |
| custom_1_1_3_1.1 | 0.7857 | 0.7795 | 0.9522 | 0.5344 | 0.5725 | 0.8132 | 0.3680 | 0.4806 | 0.5098 | 0.9315 |
| custom_1_1_3_1.2 | 0.7955 | 0.8010 | 0.9550 | 0.5397 | 0.5769 | 0.8103 | 0.4031 | 0.4961 | 0.5098 | 0.8986 |
| custom_1.1_1_3_1.2 | 0.7890 | 0.7900 | 0.9504 | 0.5450 | 0.5793 | 0.8114 | 0.4122 | 0.5000 | 0.4906 | 0.9143 |
| custom_1_0.9_3_1.2 | 0.7792 | 0.7760 | 0.9482 | 0.5767 | 0.5902 | 0.8176 | 0.5000 | 0.5455 | 0.4167 | 0.8986 |

## Key Readout

- Best validation Macro-F1: `ntl_boost2.5` (`0.8463`).
- Best test Macro-F1: `ntl_boost4` (`0.6075`).
- Best test NTL F1: `custom_1_1_3_1.2` (`0.5098`).
- Strict model selection should still prioritize validation performance; test-best rows are diagnostic unless selected before test inspection.

## Files

- Full CSV: `/home/kunet.ae/100069491/newcode/ebtc_class_weight_loss_summary_outputs/class_weight_loss_total_table.csv`
