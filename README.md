# WGAN-GP Condicional "Heavy" para Geração de Expressões Faciais

Este projeto implementa uma rede Generative Adversarial (GAN) condicional de alta performance, baseada em WGAN-GP, para gerar imagens de expressões faciais em tons de cinza de 48x48 pixels.

O modelo utiliza uma arquitetura "pesada" (`Heavy`) que incorpora várias técnicas modernas para estabilizar o treinamento e melhorar a qualidade da imagem, incluindo Normalização Espectral, Self-Attention e um design inspirado no StyleGAN.

## 🖼️ Amostras Geradas

TODO: 
(Inserir aqui uma imagem de grade de amostras geradas, por exemplo, a `samples_epoch_500.png` que o script de treinamento salva)

`![Exemplos de imagens geradas](generated_heavy/samples_epoch_500.png)`

---

## 🚀 Arquitetura e Principais Características

### 1. Gerador (`HeavyGenerator`)
* **Rede de Mapeamento (Estilo StyleGAN):** O vetor de ruído `z` e o *embedding* da classe não são alimentados diretamente na rede de convolução. Eles são primeiro processados por uma rede de mapeamento (MLP) para desacoplar o espaço latente.
* **Upsampling Progressivo:** O modelo começa com uma constante 4x4 e usa `Upsample` (Nearest/Bilinear) seguido por convoluções para aumentar a resolução (4x4 → 8x8 → 16x16 → 32x32 → 48x48).
* **Normalização por Instância (`InstanceNorm2d`):** Usada em vez de `BatchNorm` para evitar a correlação entre amostras do batch, o que é comum em modelos de estilo.
* **Self-Attention:** Uma camada de `SelfAttention` é aplicada na resolução de 16x16 para permitir que o modelo aprenda dependências de longo alcance na imagem (ex: garantir que os dois olhos façam sentido juntos).

### 2. Discriminador (`HeavyDiscriminator`)
* **Normalização Espectral (`SpectralNorm`):** Aplicada a todas as camadas convolucionais para impor a restrição de Lipschitz (1-Lipschitz), que é o núcleo do WGAN-GP, garantindo um treinamento muito mais estável.
* **Downsampling Progressivo:** Reduz a imagem de 48x48 para 3x3 através de blocos `Conv2d` com `stride=2`.
* **Self-Attention:** Também presente no discriminador (na resolução 6x6) para ajudar a reforçar relações estruturais complexas.
* **Minibatch Standard Deviation (`MinibatchStdDev`):** Uma técnica poderosa para combater o colapso de modo. Ela adiciona um canal extra às features, contendo a informação da variabilidade do batch, permitindo ao discriminador penalizar o gerador se ele produzir amostras muito similares.
* **Condicionamento por Projeção (Estilo cGAN-Proj):** O *embedding* da classe não é simplesmente concatenado à imagem. Ele é processado e concatenado a um estágio intermediário (3x3) e usado em uma camada final de classificação `Conv2d` para determinar a validade da imagem *dada* a classe.

### 3. Estratégia de Treinamento (`train_cdcgan.py`)
* **WGAN-GP:** Usa a perda Wasserstein com *Gradient Penalty* (GP) em vez da perda de entropia cruzada binária (BCELoss). Isso resulta em gradientes mais suaves e evita a saturação do discriminador.
* **Treinamento Assimétrico (`n_critic`):** O discriminador (crítico) é treinado 5 vezes (`n_critic=5`) para cada atualização do gerador, conforme recomendado para WGANs.
* **Otimizador Adam:** Utiliza os betas `(0.0, 0.9)` recomendados para WGANs.
* **Exponential Moving Average (EMA):** O script mantém uma cópia de "sombra" (EMA) dos pesos do gerador. As amostras de imagem e os checkpoints finais são salvos usando esta versão EMA, que geralmente produz resultados visualmente mais estáveis e de maior qualidade do que os pesos do gerador no último passo.

---

## 📁 Estrutura do Projeto

```

.
├── data/
│   └── fer2013/
│       └── train/
│           ├── angry/
│           ├── happy/
│           ├── sad/
│           └── ... (outras emoções)
├── checkpoints_heavy/   (Saída para checkpoints de modelo)
├── generated_heavy/     (Saída para imagens de amostra e geração)
├── train_cdcgan.py      \# Script principal para treinar o modelo
├── generate_samples.py  \# Script para gerar imagens com um modelo treinado
├── models_heavy2.py     \# Definição das classes HeavyGenerator e HeavyDiscriminator
├── datasets.py          \# Classes de Dataset (FERFolder, FERCsv)
├── utils.py             \# Funções utilitárias (salvar grade de imagens, etc.)
└── requirements.txt     \# Dependências do projeto

````

---

## 🛠️ Como Usar


### 1\. Treinamento

Para iniciar o treinamento, execute o script `train_cdcgan.py`. Todas as configurações (tamanho do lote, épocas, caminhos) estão no dicionário `CONFIG` no topo do arquivo.

  * O script criará a pasta `checkpoints_heavy` para salvar os modelos.
  * Checkpoints `.pth` completos são salvos (incluindo estados do otimizador).
  * Modelos de gerador independentes são salvos (ex: `G_epoch_500.pth` e `G_ema_epoch_500.pth`) para facilitar a geração.
  * Uma grade de amostras fixas será salva em `generated_heavy` a cada `sample_every` épocas.

### 4. Geração de Amostras

Após o treinamento, você pode usar `generate_samples.py` para gerar um grande número de imagens para cada classe.

1.  Edite o `CONFIG` em `generate_samples.py` para apontar para o checkpoint do gerador que você deseja usar (recomenda-se o `G_ema_epoch_X.pth`).

    ```python
    CONFIG = {
        'checkpoint': 'checkpoints_heavy/G_ema_epoch_500.pth', # <--- Mude aqui
        'out_dir': 'generated_heavy/epoch500_ema_inference', # <--- Mude o diretório de saída
        'num_per_class': 100,
        # ...
    }
2.  Execute o script: `python generate_samples.py`

Isso criará o diretório `out_dir` especificado e o preencherá com subpastas para cada classe, cada uma contendo `num_per_class` imagens geradas.

-----