Qualitative cases extracted from paired LLaVA/POPE logits for FastV r=0.5, seed 20260616.

| Case | idx | Gold | Unpruned | FastV | Conf delta | Residual | Question |
|:--|:--|:--|:--|:--|:--|:--|:--|
| same correct, confidence sharper | 8891 | no | no (0.585) | no (0.893) | +0.308 | 0.836 | Is there a boat in the image? |
| same correct, confidence flatter | 2808 | yes | yes (0.777) | yes (0.558) | -0.219 | 4.408 | Is there a tv in the image? |
| answer flip after pruning | 758 | yes | no (0.743) | yes (0.512) | -0.231 | 23.984 | Is there a train in the image? |
| high scalar residual | 5652 | yes | no (0.810) | no (0.570) | -0.241 | 4.238 | Is there a bowl in the image? |
