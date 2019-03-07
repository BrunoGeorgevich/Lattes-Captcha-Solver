# Lattes Captcha Solver
A simple software capable of solve Captchas from Lattes CNPq platform.

## Preparation
To run the project will be necessary download the images folder, to access the validation dataset, the model folder, to perform the CNN method, and the best template letters, to use the Naive method. The links to download the folders above mentioned can be found bellow:

- **Images** : ([Link](https://my.pcloud.com/publink/show?code=XZy02q7ZjsXlBye88GVzVNtJx57brRdoBsnV))
- **Model**  : ([Link](https://my.pcloud.com/publink/show?code=XZF02q7ZAMv4pRB08Nm3iWj8152eeLPvfuUk))
- **Templates** : ([Link](https://my.pcloud.com/publink/show?code=XZ6V2q7ZdD3TKuqvUMko9dDKuKSUF7XTbhJk))

## Requisites

* Tensorflow 1.12
* OpenCV 4.0.0
* Python 3.6.8
* Pandas 0.24.0
* Numpy 1.15.4
* Argparse 1.1

## Run
```bash
python3 main.py --method [Naive|CNN]
```

## References
### Articles

In the list bellow will be found some articles that compound the scientific foundation of the Project.

- **1.pdf**
	- *Yet Another Text Captcha Solver: A Generative Adversarial Network Based Approach*
- **2.pdf**
	- *A Survey on Breaking Technique of Text-Based CAPTCHA*
- **3.pdf**
	- *Do human cognitive differences in information processing affect
preference and performance of CAPTCHA?*
- **4.pdf**
	- *CAPTCHA – Security Affecting User Experience*
- **5.pdf**
	- *A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs*
- **6.pdf**
	- *CNN for breaking text-based CAPTCHA with noise*
- **7.pdf**
	- *Usability of CAPTCHAs or usability issues in CAPTCHA design*
- **8.pdf**
	- *New Text-Based User Authentication Scheme Using CAPTCHA*
- **9.pdf**
	- *On the Necessity of User-Friendly CAPTCHA*
- **10.pdf**
	- *Combining convolutional neural network and self-adaptive algorithm to defeat synthetic multi-digit text-based CAPTCHA*
- **11.pdf**
	- *Robustness of text-based completely automated public turing test to tell computers and humans apart*
- **12.pdf**
	- *Segmentation of connected characters in text-based
CAPTCHAs for intelligent character recognition*
- **13.pdf**
	- *CAPTCHA Recognition with Active Deep Learning*
- **14.pdf**
	- *Effects of Text Rotation, String Length, and Letter Format on Text-based CAPTCHA Robustness*
- **15.pdf**
	- *TAPCHA – An ‘Invisible’ CAPTCHA Scheme*
- **16.pdf**
	- *Text-based CAPTCHA Strengths and Weaknesses*

### Books

Some interesting books to the context of the Project.

- **Deep Learning with Python**
	- *Francois Chollet* ([Link](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438))
- Deep Learning
	- *Ian Goodfellow, Yoshua Bengio, Aaron C. Courville* ([Link](https://www.amazon.com/Deep-Learning-Ian-Goodfellow/dp/0262035618?tag=goog0ef-20&smid=A1ZZFT5FULY4LN&ascsubtag=go_1494986073_58431735035_285514469186_aud-519888259198:pla-490352386731_c_))
