## Training Accuracy

| **Model**        | **Binary**  | **Quaternary** |
|------------------|-------------|----------------|
| **CNN**          | 99\.73      | 98\.60         |
| **Feed Forward** | 99\.25      | 96\.70         |
| **Hybrid QNN**   | 99\.04      | 95\.65         |
| **QNN**          | 96\.46      | 84\.75         |

## Time

| **Model**        | **Training time \(s\)** | **Inference time \(ms\)** |
|------------------|-------------------------|---------------------------|
| **CNN**          | 11\.01                  | 0\.025                    |
| **Feed Forward** | 6\.35                   | 0\.012                    |
| **Hybrid QNN**   | 2162\.56                | 2\.773                    |
| **QNN**          | 2148\.39                | 2\.697                    |



<!-- ### Training Time

used 10000 training samples and 2000 test samples to speed up
batch size: 100
epochs: 30

### CPU
CNN  : 2.91 s (340.21 it/s) 
ANN  : 1.46 s (678.08 it/s) 
QNN8 : 151.14 s (6.55 it/s)
QNN16: 293.40 s (3.37 it/s)
QNN32: 585.18 s (1.69 it/s) 
QNN64: 1148.07 s (0.86 it/s)

### GPU
CNN  : 5.95 s (166.39 it/s) 17.75 s
ANN  : 3.91 s (253.20 it/s) 8.39 s

The quantum model is trained partially on GPU and partially on CPU so this should be disregarded
QNN8 : N/A
QNN16: N/A
QNN32: N/A (922.02 s (1.08 it/s)) 2624.83 s
QNN64: N/A

## Inference Time
CNN: 1.257 ms
ANN: 0.779 ms
QNN: 315.100 ms

*found that 10k train and 2k test gives good accuracies in 30 epochs*

Model Training Time: 10000 training samples and 2000 test samples
batch size: 100
epochs: 30

|-----------|--------------|--------------|------------------|------------------|
| Model     | Binary (CPU) | Binary (GPU) | Quaternary (CPU) | Quaternary (GPU) |
|-----------|--------------|--------------|------------------|------------------|
| CNN       |  11.01       | 16.33        |    11.06         | 17.75            |
| ANN       |    6.35      | 8.79         |     6.40         | 8.39             |
| QNN32     |  2148.39     | 2685.49      |    2169.49       | 2624.83          |
| HybridQNN |    2162.96   |              |                  |                  |
|-----------|--------------|--------------|------------------|------------------|




## Training Accuracy
%
### Binary Classification
CNN         : 99.73
Feed Forward: 99.25
Hybrid QNN32: 99.04
QNN32       : 96.46

### Quaternary Classification
CNN         : 98.60
Feed Forward: 96.70
Hybrid QNN32: 95.65
QNN32       : 84.75

## Inference time
ms/sample
### Binary Classification
CNN         : 0.025
Feed Forward: 0.012
Hybrid QNN32: 2.773
QNN32       : 2.697

### Quaternary Classification
CNN         : 98.60
Feed Forward: 96.70
Hybrid QNN32: 
QNN32       : 84.75




 -->