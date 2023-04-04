import nltk
import re
import random
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, deque
from document import training_doc

nltk.download('punkt')

class MarkovChat:
    def __init__(self, corpus=None, maxlen=2):
        self.lookup_dict = defaultdict(list)
        self._seeded = False
        self.__seed_me()
        self.maxlen = maxlen
        if corpus is not None:
            self.add_corpus(corpus)

    def __seed_me(self, rand_seed=None):
        if self._seeded is not True:
            try:
                if rand_seed is not None:
                    random.seed(rand_seed)
                else:
                    random.seed()
                self._seeded = True
            except NotImplementedError:
                self._seeded = False
    
    def add_corpus(self, corpus):
        for doc in corpus:
            preprocessed_list = self._preprocess(doc)
            pairs = self.__generate_tuple_keys(preprocessed_list)
            for pair in pairs:
                self.lookup_dict[pair[0]].append(pair[1])

    def _preprocess(self, str):
        cleaned = re.sub(r'\W+', ' ', str).lower()
        sentences = sent_tokenize(cleaned)
        tokenized = [word_tokenize(sentence) for sentence in sentences]
        return tokenized

    def __generate_tuple_keys(self, data):
        if len(data) < 1:
            return

        for sentence in data:
            for i in range(len(sentence) - 1):
                yield [sentence[i], sentence[i+1]]

    def generate_response(self, input_text):
        # Preprocess input text
        input_tokens = word_tokenize(input_text.lower())

        # Generate context for Markov Chain
        context = deque(maxlen=self.maxlen)
        for token in input_tokens:
            context.append(token)

        # Generate response using Markov Chain
        output_tokens = []
        while len(output_tokens) < 3:
            if context[-1] in self.lookup_dict:
                next_choices = self.lookup_dict[context[-1]]
                next_word = random.choice(next_choices)
                output_tokens.append(next_word)
                context.append(next_word)
            else:
                break

        # Postprocess output text
        if len(output_tokens) > 0:
            output_text = ' '.join(output_tokens).capitalize() + '.'
        else:
            output_text = "I'm sorry, I didn't understand. Could you please rephrase?"

        return output_text

# Initialize Markov Chain and add training documents
corpus = [training_doc]
my_markov = MarkovChat(corpus=corpus, maxlen=3)

# Define fallback responses
fallback_responses = [
    "I'm sorry, could you please repeat that?",
    "I'm not sure I understand. Can you provide more information?",
    "Hmm, that's interesting. Can you tell me more?",
]

# Start conversation loop
print("Hello, I'm a chatbot. How can I assist you?")
while True:
    user_input = input("> ")
    if user_input.lower() in ['exit', 'bye', 'quit']:
        print("Goodbye!")
        break
    else:
        response = my_markov.generate_response(user_input)
        if response.startswith("I'm sorry"):
            fallback_response = random.choice(fallback_responses)
            print(fallback_response)
        else:
            print(response)

