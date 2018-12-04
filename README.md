# Automatic Language Identifier
Basic Natural Language Processor  
By:  
Ahmad El-Baher `40000968` & Dylan Fernandes `40002559`

## How to Run On Lab Computers ##  
- Load right python version
`module load python/3.5.1`

## Language Prediction ##
- In `Code` directory
`python3.5 main_script.py`
- Third language : Spanish, reffered to as `ot` or `OTHER` in results

### Unigram ###
- Implementation of class in `Code/unigram.py`
- Tests in `Tests/unigram_tests.py`

### Bigram ###
- Implementation of class in `Code/bigram.py`
- Tests in `Tests/bigram_tests.py`

### Experiments ###
- Go to `Experiments` directory

#### Bigram Comparison Experiment
- Creates output files of running bigram_prev with bigram_next on sentences
- Files used for comparison of performance
- `bigram_next.py` contains class with bigram_next implementation
- `main_bigram_next_exp.py` contains logic for file IO to compare both algorithms
- Output will be placed in `Output/BigramExperiment`
- tests for class in `./Tests/bigram_next_tests.py`

#### Latin Experiment
- To be determined

## Output Files ##
- Test: Located in `Output/`

## Trained Models ##
- Located in `Trained_models`
