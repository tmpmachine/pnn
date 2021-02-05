# PNN
Probabilistic Neural Network for iris leaf species classification written on JavaScript.

I have not fully understand how PNN works hence I'm not sure if it's working correctly. Sigma is harcoded with value `0.5` and there's still few things left to do (I have not find any reference on how to determine optimal sigma value beside guessing and testing between scoped interval).

## References
- [PNN Tutorial](https://www.cse.unr.edu/~looney/cs773b/PNNtutorial.pdf)
- [Using Probabilistic Neural Networks for Handwritten Digit Recognition](https://scialert.net/fulltext/?doi=jai.2011.288.294)
- [Confusion Matrix](https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826#:~:text=To%20calculate%20accuracy%2C%20use%20the,or%20(1%2DAccuracy))

## Testing
1. Open `index.html`.
2. Select training file inside `data-samples` folder.
3. Select testing file inside `data-samples` folder.
4. See browser `console` for the classification result.
