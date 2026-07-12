# Local Model Manifest

Model checkpoint binaries are intentionally ignored by Git. Keep them locally or publish them through the artifact store used for the paper release.

| file | bytes | sha256 |
| --- | ---: | --- |
| arxiv-6_layer-model-jigsaw_coverage_final_ablation_final_seed7101_best_fullcov.pth | 15784066 | 1f41298d2246327e6c2e0f6588a0d1c2660936e67a6b05033374f3d2b98af7c5 |
| arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6_best_fullcov.pth | 15780562 | 48aa1b9525b7e933c978b9efc977f97b0ea4b31401b26dd0cddf8492f82f21c1 |
| arxiv-6_layer-model-jigsaw_coverage_v7_continue_from_v6.pth | 15779898 | 034c0953900d3fb3e3a0a34f1dde1a5a4a9cf236fd2f44f054321bf301abf362 |
| arxiv-6_layer-model-jigsaw_v1.pth | 15645602 | 8304792e9b9a297d53567dedd8649d1ba30ca6ad6a53de0388f6a8cc4c836133 |
| arxiv-6_layer-model-jigsaw_v2.pth | 15777173 | c2dfc7a04441d3e2b7a6bd73d52dc01ed38d953772b1ad53f5c7e96ef435915c |
| arxiv-6_layer-model-jigsaw_v3.pth | 15777173 | 96cd9e767c1ba13f6fc149f99831f089b00b9ad009c34e409f2b7fff34949dbc |
| cora-6_layer-model-jigsaw_v1.pth | 33222043 | 5064ac2c3be222c55c15a2ec47b3df66bb3db2cc8daac63a9ce40a36b8dabd1e |
| cora-6_layer-model-jigsaw.pth | 24565083 | ecffcca91d90292cdc63ae226e301eb8eb026b488ce783d195fa6903ad12a4d4 |
| mag-6_layer-model-jigsaw.pth | 139856061 | 6da2e1dc526c80d6af7dd8fa5f78f29b762453e12fd76fd459be46ce8de46fdd |
| training_log_arvix.txt | 18167 | 58aecaf21425713f9ddcd963747f6682160d49d5f366ab194f4bd5a11834a1c0 |
| training_log_cora.txt | 15592 | 953c4dc9a6e6a0ca1a1cfdf0b41ca3971454e4b8411df21f8e9a4d8f79575741 |
| training_log_mag.txt | 1062424 | 60e6863174073d191a36238af5c35d89fed3ffb6cf3a77403bd0085088d1800e |

## Overlap-Trained GraphSAGE Checkpoints

These post-submission checkpoints are stored under `runs/overlap_models/` and are packaged by
`scripts/lightning_production_benchmark.py` when present.

| file | bytes | sha256 |
| --- | ---: | --- |
| runs/overlap_models/cora/models/cora-6_layer-model-graphsage_graphsage_final_loss_cora_seed7202_overlap_topk10_live20_best_fullcov.pth | 24561794 | 2c18b5cee4f08e28983894942f7eba4747f129817ab7d73a6d5ec37e85f1feec |
| runs/overlap_models/cora/models/cora-6_layer-model-graphsage_graphsage_final_loss_cora_seed7202_overlap_topk10_live20.pth | 24562438 | c9515e02b47b8b58e12466882e526ffc641f57cc02694399357cc91a7f71bcd3 |
| runs/overlap_models/arxiv/models/arxiv-6_layer-model-graphsage_graphsage_final_loss_arxiv_seed7202_overlap_topk50_live64_best_fullcov.pth | 15773914 | 7a50dc1cbf6a760e33c9a1db4b9f6ce02d38c928ed055848e4d249db6aecd276 |
| runs/overlap_models/arxiv/models/arxiv-6_layer-model-graphsage_graphsage_final_loss_arxiv_seed7202_overlap_topk50_live64.pth | 15774622 | d7f604f255047dd2a75f52864b45e10cc75bd4cc7733fac936732c42cd6ba75c |
