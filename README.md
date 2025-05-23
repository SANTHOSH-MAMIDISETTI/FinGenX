# FinGenX: Generative AI Framework for Synthetic Financial Data  
*"Bridging Data Scarcity and Privacy in Financial AI"*  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)  

## Table of Contents  
- [Key Features](#key-features)  
- [Architecture Overview](#architecture-overview)  
- [Installation](#installation)  
- [Model Architectures](#model-architectures)  
- [Performance Benchmarks](#performance-benchmarks)  
- [Applications](#applications)  
- [Team Members](#team-members)  
- [License](#license)  
- [Contact](#contact)  

## Key Features  
**Hybrid Generative Framework** combining GAN-Transformer and Denoising Diffusion Probabilistic Models (DDPM) for:  
- 📈 Temporal pattern preservation in financial time-series  
- 🔒 GDPR-compliant data generation through noise injection  
- 🧩 Simultaneous handling of tabular (credit data) and sequential (market data) formats  
- 🎯 Five-dimensional evaluation: Fidelity (Kolmogorov-Smirnov Test), Utility (ML Performance), Privacy (DCR), Synthesis (Novelty), Temporal Correlation (Autocorrelation)  

## Architecture Overview  
![System Architecture](https://via.placeholder.com/800x400.png?text=GAN-Transformer+%2B+Diffusion+Model+Architecture)  
*Dual-model architecture enabling high-fidelity generation and privacy-preserving synthesis*

## Installation  
```
git clone https://github.com/SANTHOSH-MAMIDISETTI/FinGenX.git  
cd FinGenX  
```

## Model Architectures  
### GAN-Transformer Hybrid  
```
class GANTransformer(nn.Module):
    def __init__(self, attention_heads=8, num_layers=6):
        self.generator = TransformerEncoder(
            d_model=512, nhead=attention_heads, num_layers=num_layers
        )
        self.discriminator = TransformerEncoder(
            d_model=512, nhead=attention_heads, num_layers=num_layers
        )
```

### Denoising Diffusion Model  
```
class FinancialDiffusion(nn.Module):
    def forward_process(self, x0, t):
        sqrt_alpha = torch.sqrt(self.alphas[t])
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas[t])
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * torch.randn_like(x0)
```

## Performance Benchmarks  
| Metric              | GAN-Transformer | Diffusion Model |  
|---------------------|-----------------|-----------------|  
| **Fidelity (KS Test)** | 0.95            | 0.63            |  
| **Utility (AUC)**      | 0.92            | 0.68            |  
| **Privacy (DCR)**      | 0.75            | 0.92            |  
| **Training Time**      | 8.2h            | 14.5h           |  

## Applications  
- 🏦 Synthetic credit portfolios for risk modeling  
- 📉 Market simulation under different regimes  
- 🕵️ Adversarial fraud detection training  
- 📊 Privacy-safe financial analytics  

## Team Members

| Name                | Roll Number        |
|---------------------|--------------------|
| Amrithnarayana K    | AM.EN.U4AIE21012   |
| Santhosh M          | AM.EN.U4AIE21042   |
| Nandakishor P       | AM.EN.U4AIE21045   |
| Navneeth Krishna    | AM.EN.U4AIE21047   |

## License  
Distributed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.  

## Contact  
**Project Lead**: [Santhosh Mamidisetti](https://linkedin.com/in/santhosh-mamidisetti)  
**Institutional Support**: Amrita Vishwa Vidyapeetham, Amritapuri Campus
