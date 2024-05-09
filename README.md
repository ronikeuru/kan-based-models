# Kolmogorov-Arnold Network (KAN) Based Models

Testing different deep learning model architectures with Kolmogorov-Arnold Networks.

Currently `main.py` trains a Convolutional KAN model with MNIST dataset.

## How To

Run the following commands:

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python src/main.py

## References

- [Original Paper](https://arxiv.org/abs/2404.19756)
- [Original Repo](https://github.com/KindXiaoming/pykan/)

This work was inspired and based on following efficient implementations:

- [Efficient Implementation 1.](https://github.com/Blealtan/efficient-kan)
- [Efficient Implementation 2.](https://github.com/Indoxer/LKAN)

More information on the topic:

- [About B-Splines](https://en.wikipedia.org/wiki/B-spline)
