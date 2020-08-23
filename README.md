# transcoder

Dockerized API version of TransCoder, based on the [PyTorch original implementation](https://github.com/facebookresearch/TransCoder).
The code implements what's described in the paper [Unsupervised Translation of Programming Languages](https://arxiv.org/pdf/2006.03511.pdf).


## Requirements

### API

- a machine having a CUDA-compatible NVIDIA GPU;
- `nvidia-docker` installed and running;

### Demo

- Docker


## How to run

### API

At first you need to **download the models weights**. In order to do that, run this command from the root folder of this project:
```bash
make -f Makefile.api download-models-weights
```

In your GPU-enabled machine, just run:

```bash
make -f Makefile.api build run
```

### Demo (in local)

- To run it in local, make sure you have an environment variable called `TRANSCODER_API_HOST` set like `http://your-host:8080`;
- Execute:
```bash
make -f Makefile.demo build run
```

## Deployment

At the moment it's not possible to deploy the API in Convox due to the lack of racks with GPU nodes in EKS.
To deploy the demo, run from the root folder of this project (assuming you have Convox correctly configured):

```bash
convox deploy --no-cache -r lab-irl
```


## References
This Code was used to train and evaluate the TransCoder model in:

[1] M.A. Lachaux*, B. Roziere*, L. Chanussot, G. Lample [Unsupervised Translation of Programming Languages](https://arxiv.org/pdf/2006.03511.pdf).

\* Equal Contribution

```
@misc{facebookresearch,
  title={TransCoder},
  author={Lachaux, Marie-Anne and Roziere, Baptiste and Chanussot, Lowik and Lample, Guillaume},
  year={2020},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/facebookresearch/TransCoder}},
  commit={deeedd00921ac9db9a201b858bbecb40994f2e54}
}

@article{lachaux2020unsupervised,
  title={Unsupervised Translation of Programming Languages},
  author={Lachaux, Marie-Anne and Roziere, Baptiste and Chanussot, Lowik and Lample, Guillaume},
  journal={arXiv preprint arXiv:2006.03511},
  year={2020}
}
```

## License

This software is under the license detailed in the Creative Commons Attribution-NonCommercial 4.0 International license. See LICENSE for more details.
