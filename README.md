Ai Project 
Group Members: Muhammad Yahya 37236
               Ayaan Khan 37274
			   Fahad Ahmed 37258
---
### Code structure:
- `train.py` Train on ITS/OTS of RESIDE dataset, including validation.  
- `test.py` Test on SOTS of RESIDE dataset, evaluation metrics are PSNR and SSIM, dehazed images could be saved in folder `test_results`.
- `dehaze.py`  Dehazing demo, dehazed images saved in folder `demo_results`.
- `net.py` Network, modified baseline-AOD on the basis of original AOD-Net, introducing attention mechanism (PA, SA), etc. 
- `dataloader.py` Preprocess data.
- `common.py` Hyperparameters (could be packaged to `args`)
- `orig_net.py` Original AOD-Net.

