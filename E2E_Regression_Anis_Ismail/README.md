<!--
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!--[![MIT License][license-shield]][license-url] -->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Applying Deep Learning for Mass Regression of Boosted Top Quarks Particles at the CMS Experiment at CERN</h3>

  <p align="center">
My summer project as a Deep Learning Intern at the Machine Learning For Science Group (ML4SCI) part of Google Summer of Code 2021.
    <br />
    <a href="https://github.com/ML4SCI/E2E/tree/main/E2E_Regression_Anis_Ismail"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/ML4SCI/E2E/tree/main/E2E_Regression_Anis_Ismail">View Demo</a>
    ·
    <a href="https://github.com/ML4SCI/E2E/issues">Report Bug</a>
    ·
    <a href="https://github.com/ML4SCI/E2E/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
   <!-- <li><a href="#license">License</a></li> -->
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project is part of my work as a Deep Learning Intern at the Machine Learning For Science Group (ML4SCI) part of Google Summer of Code 2021.

add logo gsoc and cms and ml4sci
[![Product Name Screen Shot][product-screenshot]](https://example.com)

You can learn more about the project [here]().

### Built With

* [Pytorch](https://pytorch.org/)
* [NumPy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* All required packages can be installed as follow:
  ```sh
  pip install -r requirements.txt
  ```
* **Please note that this project requires a GPU for training**
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ML4SCI/E2E.git
   ```
2. Change to the project repositry:
   ```sh
   cd E2E_Regression_Anis_Ismail/

   ```

3. Install required packages
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

To start training the model, run the following command:
   ```sh
   python main.py
   ```
You can modify the experiment parameters in the experiment.json file:
   ```json
{
  "batch_size": 1024,
  "epochs": 50,
  "load_epoch": 0,
  "lr": 1e-3,
  "resblocks": 3,
  "input_channels": 3,
  "fmaps": [
    16,
    32
  ],
  "is_cuda": 1,
  "run_logger": 1,
  "expt_name": "TopGun_scaled-target&input-500-0.02-0.2-1_lr_scheduled-1e-3",
  "save_path": ".",
  "data_path": ".",
  "channel1_scale": 0.02,
  "channel2_scale": 0.2,
  "channel3_scale": 1.0,
  "seed": 0
}
   ```

<!-- ROADMAP -->
## Roadmap


See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE 
## License

Distributed under the MIT License. See `LICENSE` for more information.
-->


<!-- CONTACT -->
## Contact

Your Name - [Anis Ismail](https://linkedin.com/in/anisdimail) - anis[dot]ismail[at]lau[dot]edu

Project Link: [https://github.com/ML4SCI/E2E/tree/main/E2E_Regression_Anis_Ismail](https://github.com/ML4SCI/E2E/tree/main/E2E_Regression_Anis_Ismail)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Davide Di Croce](https://ch.linkedin.com/in/davide-di-croce-2840961a3)
* [Sergei Gleyzer](https://www.linkedin.com/in/sergei-v-gleyzer-a72b772)
* [Darya Dyachkova](https://www.linkedin.com/in/darya-dyachkova)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ML4SCI/E2E.svg?style=for-the-badge
[contributors-url]: https://github.com/ML4SCI/E2E/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ML4SCI/E2E.svg?style=for-the-badge
[forks-url]: https://github.com/ML4SCI/E2E/network/members
[stars-shield]: https://img.shields.io/github/stars/ML4SCI/E2E.svg?style=for-the-badge
[stars-url]: https://github.com/ML4SCI/E2E/stargazers
[issues-shield]: https://img.shields.io/github/issues/ML4SCI/E2E.svg?style=for-the-badge
[issues-url]: https://github.com/ML4SCI/E2E/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/anisdimail
<!--[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt -->
