# Detecting COVID-19 in Lungs using CNN

Initially, the model resulted in only **46% accuracy**. Therefore, I decided  to experiment with different strategies for increasing accuracy:

| Methodology                           | Accuracy (%)    | Description                                                              |
|---------------------------------------|----------------|--------------------------------------------------------------------------|
| Initial CNN Model                     | 46%            | Basic CNN implementation.                                                |
| Transfer Learning (ResNet)            | 82%            | Used transfer learning with ResNet.                                       |
| Hybrid CNN + ResNet                   | 85%            | Combination of self-made CNN with ResNet.                                 |
| Learning Rate: 0.0001, betas=(0.8, 0.999), weight decay=1e-5 | 87.291%        | Tuned learning rate and optimizer parameters. Used Swish activation.     |
| With Batch Normalization              | 86.25%         | Introduced batch normalization layers.                                    |
| With Learning Rate Scheduler          | 85.625%        | Implemented a learning rate scheduler to adjust the LR during training.  |
| Epochs Increased (10 to 20) + L1 Loss | 86.875%            | Increased epochs and added L1 regularization for better performance.      |

