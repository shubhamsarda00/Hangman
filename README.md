# Hangman

## Overview
This project aims to solve the classic game of Hangman using two different strategies: a probability-based greedy approach and a neural network-based method. The model is trained using a dataset of 250,000 words.

## Input
**Training Set:** 250,000 words from a dictionary dataset.

## Strategies for the problem

### 1)	Greedy/Probability based Solution:

This method relies on string matching and frequency analysis to make informed guesses. Here's a breakdown of the steps:

**1) Word Length Filtering:** Filter the dictionary to only include words that match the length of the target word.

**2) Pattern Matching:** Further filter this list to include only those words that match the currently known letter positions.

**3) Frequency Distribution:** For each unknown letter position, create a frequency distribution by checking which letters appear most often at that position in the filtered list. Then, create individual probability distributions for each unknown index.

**4) Guess Strategy:**

Approach A: For each distribution, pick the letter with the highest individual probability.

Approach B: Normalize and sum all distributions to form a global probability distribution and guess the most probable letter overall.

These strategies are implemented in the functions guess1, guess2, and guess3. However, they achieve only ~18% win rate on the test set. This approach could be further enhanced by incorporating n-grams, which was not explored due to time constraints.

### 2)	Neural Network Based Approach:

This approach uses a machine learning model, specifically an LSTM network with attention, to learn the optimal guessing strategy.The challenge here is not developing the neural network model but to forumulate correct input and output for the same. In order to learn the hangman strategy, we try to emulate the exact game scenario to generate optimal input and outputs.

#### **Problem Formulation:**

**Input:** 

1) The current state of the partially masked word  (e.g., _ p p _ e).

2) The set of previously guessed letters.

**Output:** - The next most probable unguessed letter from the target word.

**Data Generation**

To train the model:

1) We simulate Hangman games using the training dictionary.

2) For each word, the game is played until it's solved or 6 incorrect guesses are made.

3) Each game produces multiple game states (input-output pairs), significantly increasing training data volume as each word can provide at least 6 game states (input-output pair).

**Input Encoding**

- **Part 1: Word Representation:**

  - One-hot encoded representation of the partially guessed word.

  - Shape: (maxLen, 28), where maxLen is the maximum word length in the dataset.

  - The 28 possible values per position include 26 letters (a–z), "_" (unguessed), and "." (padding).

- **Part 2: Previous Guesses:**

  - Binary vector of shape (1, 26) representing whether each letter has been guessed.

**Output:**

A probability distribution over the remaining unguessed letters.

We create a classification model using LSTM+attention to predict the guess letter. 

**Model Architecture**

- **Input 1 (One hot encoded Word):**

    - Passed through two LSTM layers with an attention layer in between.

    - Followed by a dense (fully connected) layer.

- **Input 2 (Previous Guesses):**
 
    - Processed through fully connected layers.

- **Merged Output:**

    - The outputs from both inputs are concatenated.

    - Passed through two more dense layers.

    - Final layer is a softmax producing a (1, 26) probability distribution over letters.

- **Model Note:**
    - maxLen = 29 in the current implementation.
    - Experiments were done with deeper networks as well. They however did not give very good results due to overfitting (40% win rate).

<img width="979" height="713" alt="image" src="https://github.com/user-attachments/assets/57a01dbb-1a9a-4744-a71e-e5adc1d04544" />

 
**Model Training:**

- **Dataset Split:**

  - Train set: 210,000 words

  - Test set: 17,300 words

- **Training Strategy:**

  - Since each word generates multiple game states, the actual number of training pairs is significantly higher.

  - Due to high computational cost (~2 days per epoch), training was limited to a few epochs.

  - A trainOnTest() function was used to fine-tune the model on the test set.

- **Results:**

  - Final model achieved 51.7% win rate after 3 epochs.

  - Further training is expected to yield better performance.



**Code Structure:**

- **Training Code:** Found under the "Training" section.

- **Inference Code:** Under "Final Submission", which loads the final trained model from model3_3.h5.


**Library Requirements:**

1)	Numpy
2)	Keras
3)	keras_self_attention (Installation: pip install keras_self_attention): this is a wrapper library to add attention layers to keras models. (https://pypi.org/project/keras-self-attention/)
