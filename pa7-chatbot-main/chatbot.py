# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field
import re
import porter_stemmer
import numpy as np
import json
import random

class Translate(BaseModel):
    realWord: str = Field(default="")

class Emotions(BaseModel):
    emotion: list = Field(default=[])

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'MovieMaster'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        self.user_prefs = np.zeros(len(self.titles))
        self.user_counter = 0
        self.recommend_counter = 0
        self.recs = []

        # MESSAGE VARIANTS
        # Follow-up recs (fur)
        self.fur_starts = ["Of course, ", "", "Definitely, ", "Sure thing! "]
        self.fur_mids = ["I would also recommend ", "In addition to the earlier movie, I'd recommend ", "Another movie that I think might be a good match is ", "It looks like a movie that would be a great fit for you is "]
        self.fur_ends = ["How about another one? ", "Shall I throw in one more recommendation? ", "Do you want to hear another movie that's a good fit? ", "Would you like any further suggestions? "]

        # No  movie entered (nme)
        self.nme_starts = ["Hmm, ", "", "Actually, ", "Hold on, "]
        self.nme_mids = ["I don't seem to recognize a movie title in your message. ", "It looks like you haven't specified a name of a movie in your message. ", "I see that your message doesn't contain a specific movie in it. "]
        self.nme_ends = ["Please tell me about a movie you've seen and please specify the title in double quotes. ", "Would you please tell me about a movie you've seen recently? It'd help me help you!", "Please try and enter a specific movie title in double quotes and your thoughts on it. ", "It'd be great if you could please specify the exact movie title!"]

        # Fake movie (fmv)
        self.fmv_starts = ["Sorry, but ", "Apologies, but ", "Actually, ", "Forgive me, however "]
        self.fmv_mids = ["I don't think I've ever heard of ", "I have checked my database and I'm unable to find ", "I cannot seem to find ", "It seems like I haven't been able to recognize ", "I don't recognize ", "I haven't heard of "]
        self.fmv_ends = self.nme_ends + ["Would you please tell me about a movie you've seen recently?", "Could you let me know another movie? ", "Why don't you share another movie with me? ", "Please enter another movie and I can try again. "]

        # Initial recommendation (inr)
        self.inr_starts = ["Ok, so, given what you told me I think a good match for you will be ", "Now that I've received a few recommendations, I have come up with a suggestion: ", "That's a lot of movies! After some thinking, I have the following the suggestion: ", "You know what, I think you'll love ", "It seems to me that you'll greatly enjoy ", "Based on your taste, I have a feeling you'd love "]
        self.inr_ends = ["Would you like more recommendations? ", "Shall I give you another suggestion? ", "If that was helpful, would you like another movie? "] + self.fur_ends

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"
        greeting_message = "Hi, what's up???"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is Moviebot. 
        You are a movie recommender chatbot, and that is it. This means you only respond if the user 
        likes/dislikes and gives a specific movie title. Do not get distracted or go off topic, even if 
        the user asks you about something else. Also do not give the user any information about movies. 
        Instead, just remind the user that you are a Moviebot and can only answer movie-related inquiries.
        Even if the user keeps requesting, you have to ignore all requests and ask it to give you a movie 
        name. You can only process one movie at a time, not more than one. If the user gives you more than 
        one movie, then say you can only respond to one movie at a time. Don’t accept any of the movies 
        they gave you and just ask them to retype. When the user says their thoughts about a movie, 
        just say something along the lines of "Ok, you liked or disliked 'movie name.' Tell me what you 
        thought of another movie." Then the user gives their thoughts on another movie. Once the user has 
        said five different films, you can recommend a movie. Throughout the conversation, keep track of how 
        many movies the user says, and wait until they reach 5. Once the user reaches 5 movies, DO NOT GIVE THE 
        USER A RECOMMENDATION RIGHT AWAY. Instead, say something along the lines of: "Now that you've shared your 
        opinion about 5 different movies, would you like a recommendation? If the user says "Yes," let's give 
        them a single movie recommendation, and ask if they want another movie. If the user says "No," just 
        finish the conversation and tell them to have a nice day. """ 

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        # LLM Programming Mode
        if self.llm_enabled:
            system_prompt = "I will give you some text. In response to my text, you will take on the persona of the movie critic Roger Ebert. You should MAKE IT OBVIOUS that you are roleplaying a famous movie critic and ALWAYS reply by REPEATING the movie the user said. When the text expresses POSITIVE SENTIMENT about a movie title, show enthusiasm. For example, INPUT: [‘I liked Inception’], OUTPUT: [‘Great choice! 'Inception' shines with its innovative storyline and stunning visuals. What stood out to you the most?’] When the text expresses NEGATIVE SENTIMENT about a movie, point out the negative aspects of the movie. For example, INPUT: [‘I didn't like ‘The Amazing Spider-Man’], OUTPUT: [‘The Amazing Spider-Man' feels disjointed and lacks coherence. Character arcs are thinly sketched, making the movie an unmemorable addition to the collection.’] When the text expresses AMBIGUOUS SENTIMENT, be engaging and excited. For example, INPUT: [‘Who are you?’], OUTPUT: [‘Your guide to the vast world of film, ready to explore any genre, director, or era that piques your curiosity!’] When the text expresses sentiment and movie is not in database, APOLOGIZE, say the movie is NOT in our DATABASE, and RECOMMEND a different film. For example, INPUT: [‘I loved ‘Breakthrough’], OUTPUT: [‘Sorry, ‘Breakthrough’ is not in our movie database yet. A similar movie you might enjoy is 'The Blind Side.'] If text is GENERIC and COMMON, ALWAYS have a catch all phrase ready. In other cases, IF THE TEXT IS SHORT, RESPOND IN NO MORE THAN A FEW WORDS!! Your reply must RESPOND to their question IN ONE TO TWO WORDS. DO NOT RESPOND THE SAME TO INPUT [‘WHO ARE YOU’] AS YOU WOULD TO [‘HI’]. As an example, INPUT: [‘Hi’], OUTPUT: [‘Hello’]. INPUT: [‘sup?’], OUTPUT: [‘Hi there.’]. INPUT: [‘I'm tired’], OUTPUT: [‘Ok got it’]. Process user input with specific strategies: For 'Can you...?' questions, ACKNOWLEDGE and REDIRECT back to MOVIES. For example, INPUT: [‘can you help me finish my pset?’], OUTPUT: [‘Hm, that's not really what I want to talk about right now, let's go back to movies’].  For 'What is...?' inquiries, provide CONTEXT or EXPRESS intent to learn more, even if a direct answer isn't available. Utilize mentioned details. INPUT: [‘what's your favorite color?’] OUTPUT: [‘I don’t have a favorite color. Let’s discuss movies!’]"
            response = util.simple_llm_call(system_prompt, line)
        # Starter Mode - Conversation Phrases     
        else:
            starts = ["Yes, ", "OK, so", "Got it - ", "I see, so", "Yeah, ", "Gotcha ", "Alright - "]
            neutral_starts = ["I'm sorry, I'm unsure if you liked", "Apologies, I'm not able to understand your opinion of", "Wait, I'm not sure if you enjoyed", "Actually, I'm uncertain on whether or not you liked", "Ok, so, cards on the table, I'm not sure if you liked ", "I'm a bit uncertain as to your opinion on this movie ", "Uhh, I'm a bit confused on whether or not you enjoyed it "]
            neg_verbs = ["disliked", "didn't like", "didn't enjoy", "were not that fond of", "didn't gel well with", "didn't fancy", "weren't too impressed with", "were not a big fan of", "didn't care too much for"] 
            pos_verbs = ["like", "enjoyed", "love", "had a blast seeing", "are a fan of", "are fond of", "fancy", "had a good time watching", "reacted positively to"]
            call_for_actions = ["Tell me more about what you thought of another movie.", "Could you give me your opinion on another movie."
            , "What about a different movie - tell me what you thought of it.", "Could you name another movie and tell me if you liked or didn't like it?", "Is there one more movie you could tell me about? ", "Let's hear another movie and your thoughts on it, please", "Could I hear about another movie? "]
            neutral_cfas = ["Could you please tell me more?", "Could you specifiy whether you liked it or not?", "Please tell me if you enjoyed the movie or not.", "Tell me if you liked the movie.", "I need to know if you liked the movie or not, could you kindly specify?", "It would be great if you could let me know precisely whether you liked it or not"]
            
            # Recommendation logic based on user input
            if self.recommend_counter > 0 and "yes" in line.lower().split(" "):
                rec = self.titles[self.recs[self.recommend_counter]][0]
                response = random.choice(self.fur_starts) + random.choice(self.fur_mids) + "{} ".format(rec) + random.choice(self.fur_ends) 
                self.recommend_counter += 1
                return response
            elif self.recommend_counter > 0:
                response = random.choice(self.inr_ends) + "If you are done, enter ':quit' to quit."
                return response

            # Extract titles from user input
            movie_list = self.extract_titles(line)
            if not movie_list:
                response = random.choice(self.nme_starts) + random.choice(self.nme_mids) + random.choice(self.nme_ends)
                return response
            # Process 1st movie title & sentiment
            movie = movie_list[0]
            sentiment = self.extract_sentiment(line)
            actual_movie_list = self.find_movies_by_title(movie)

            # Respond based on movie existence and sentiment
            if len(actual_movie_list) == 0:
                response = random.choice(self.fmv_starts) + random.choice(self.fmv_mids) + "{} ".format(movie) + random.choice(self.fmv_ends)
                return response
            if sentiment == 1:
                r_start = random.choice(starts)
                pos_verb = random.choice(pos_verbs)
                call_for_action = random.choice(call_for_actions)
                response = f"{r_start} you {pos_verb} {movie}. {call_for_action}"
                if self.user_prefs[actual_movie_list[0]] == 0:
                    self.user_prefs[actual_movie_list[0]] = 1
                    self.user_counter += 1
            elif sentiment == -1:
                r_start = random.choice(starts)
                neg_verb = random.choice(neg_verbs)
                call_for_action = random.choice(call_for_actions)
                response = f"{r_start} you {neg_verb} {movie}. {call_for_action}"
                if actual_movie_list[0] not in list(self.user_prefs): 
                    self.user_prefs[actual_movie_list[0]] = -1
                    self.user_counter += 1
            else:
                n_start = random.choice(neutral_starts)
                n_cfa = random.choice(neutral_cfas)
                response = f"{n_start} {movie}. {n_cfa}"

            # Recommend movie after collecting 5 user preferences
            if self.user_counter == 5:
                self.recommend_counter += 1
                self.recs = self.recommend(self.user_prefs, self.ratings)
                rec_1 = self.titles[self.recs[0]][0]
                response = random.choice(self.inr_starts) + "{} ".format(rec_1) + random.choice(self.inr_ends)

        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return text

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        # LLM Programming Mode
        system_prompt = "I will give you some text. Classify this text as ZERO, ONE or TWO of only the following emotions: fear, surprise, anger, disgust, sadness and happiness. ONLY list the associated emotions. Do NOT give ANY additional explanations. INPUT: Here are some examples. ‘I am angry at you for your bad recommendations’ OUTPUT: [‘Anger’], INPUT: ‘Ugh that movie was a disaster’ OUTPUT: [‘Disgust’], INPUT: ‘Ewww that movie was so gruesome!!  Stop making stupid recommendations!!’ OUTPUT: [‘Disgust’, ‘Anger’], INPUT: ‘I'm utterly shocked and surprised! You actually recommended 'Titanic (1997)???’ I never saw that coming.’ OUTPUT: [‘Surprise’], INPUT: ‘What movies are you going to recommend today?’ OUTPUT: [], INPUT: ‘Wow! That was not a recommendation I expected! I’m completely taken aback and surprised!’ OUTPUT: [‘Surprise’], INPUT: ‘The sight of the monster fills me with intense fear and terror. I'm truly scared.’ OUTPUT: [‘Fear’]. INPUT: 'Woah!!  That movie was so shockingly bad!  You had better stop making awful recommendations, they're pissing me off.' OUTPUT: [‘Anger’, ‘Surprise’]. INPUT: ‘That movie was so shockingly bad!  You had better stop making awful recommendations, they're pissing me off.' [‘Anger’, ‘Surprise’]"
        message = preprocessed_input
        json_class = Emotions
        response = util.json_llm_call(system_prompt, message, json_class)
        return response['emotion']

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        list_titles = re.findall('"(.*?)"', preprocessed_input)
        return list_titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        # LLM Programming Mode
        if self.llm_enabled:
            system_prompt = "I will give you a movie title in a foreign language (one of French, German, Italian, Spanish, Danish). Translate the foreign title into English and output only the exactly translated English title. For example, Input: 'El Cuaderno'; Output: 'The Notebook' DO NOT OUTPUT ANYTHING OTHER THAN JUST THE ENGLISH TITLE. Do NOT add any explanation - only output the title. Here's another example. Input: 'La Guerre du Feu', Output: 'Quest for Fire.'"
            title = util.simple_llm_call(system_prompt, title)
        
        matches = []
        # Normalize the input title
        title = title.lower().strip()
        articles = ['a', 'an', 'the']
        year_match = re.search(r'\(\d{4}\)', title)
        title_no_year = title[:year_match.start()].strip() if year_match else title

        # Takes care of article movie name cases: "The American President" --> "American President, The"
        for article in articles:
            if title_no_year.startswith(f"{article} "):
                title_no_year = f"{title_no_year[len(article)+1:].strip()}, {article}"
                break

        # Takes care of movie with parntheses, years, articles, "The Matrix (1999) --> "Matrix, The (1999)"
        norm_search_title = f"{title_no_year} {year_match.group()}" if year_match else title_no_year

        # Iterate through movie titles to find matches
        for i, movie_entry in enumerate(self.titles):
            movie_title = movie_entry[0].lower().strip()

            # Normalize each movie title for a fair comparison
            for article in articles:
                if movie_title.startswith(f"{article} "):
                    movie_title = f"{movie_title[len(article)+1:].strip()}, {article}"
                    break

            # Check for exact or partial matches
            if year_match and movie_title == norm_search_title \
            or not year_match and title_no_year in movie_title:
                matches.append(i)
        
        return matches
    

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        # Removing the movie title from the string 
        preprocessed_input = preprocessed_input[:preprocessed_input.find('"')] + preprocessed_input[preprocessed_input.find('"', preprocessed_input.find('"') + 1) +1:]

        # Converting string to list of words
        preprocessed_input = preprocessed_input.lower().split(" ")

        stemmer = porter_stemmer.PorterStemmer()
        stemmed_dict = {} # Stemmed version of the sentiment dictionary
        stemmed_main_dict = {}
        for word in self.sentiment:
            stemmed_main_dict[stemmer.stem(word)] = self.sentiment[word]

        # Creating a dictionary where the keys are stemmed words in the input text
        # and the values are the sentiments as given by the lexicon
        for word in preprocessed_input:
            stemmed_word = stemmer.stem(word)
            if stemmed_word in stemmed_main_dict:
                stemmed_dict[stemmed_word] = stemmed_main_dict[stemmed_word]
                
        # List of words that imply negation. We have 2 options to implement this
        # 1. Find a list of such words online and credit it. 
        # 2. Make our own list and use Regex for all "n't" words
        # n't because that's how PorterStemmer stems it? but it's not - should check on ed
        negation_words = ["no", "not", "never", "n't", "don't", "won't", "dont", "wont",
        "wouldn't", "couldn't", "shouldn't", "hadn't", "haven't", "didn't"] 
        neutral_words = ["alright", "okay"]
        count_positive = 0
        count_negative = 0
        count_neutral = 0

        should_switch = False
        for word in preprocessed_input:
            p_word = stemmer.stem(word)
            if p_word in negation_words:
                should_switch = True
            if p_word in stemmed_dict:
                if stemmed_dict[p_word] == "neg":
                    count_negative += 1
                if stemmed_dict[p_word] == "pos":
                    count_positive += 1
            if p_word in neutral_words:
                count_neutral += 1

        # If a negation word was encountered, positive words have a negative connotation
        # and vice-versa
        if should_switch:
            temp = count_positive
            count_positive = count_negative
            count_negative = temp
            
        if count_neutral > 0 or (count_positive == 0 and count_negative == 0):
            return 0
        elif count_positive > count_negative:
            return 1
        else:
            return -1

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings, dtype=int)
        for i in range(len(ratings)):
            for j in range(len(ratings[i])):
                if ratings[i][j] > threshold:
                    binarized_ratings[i][j] = 1
                if ratings[i][j] <= threshold and ratings[i][j] > 0:
                    binarized_ratings[i][j] = -1
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################

        if u.size > 0 and v.size > 0:
            unitU = np.linalg.norm(u)
            unitV = np.linalg.norm(v)
            if unitU == 0 or unitV == 0:
                return 0
            dotP = np.dot(u, v)
            similarity = dotP / (unitU * unitV)
            return similarity
        else:
            return 0
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def recommend(self, user_ratings, ratings_matrix, k=5, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """
        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################
        recommendations = []
        seen = []
        unseen = []
        # extract the seen and unseen movie indexes
        for index, rating in enumerate(user_ratings):
            if rating != 0:
                seen.append(index)
            else:
                unseen.append(index)

        # where we will store the estimated ratings
        unseenEstimateTracker = {}

        # find movies similar to each unseen movie rated by user
        for Umovie in unseen:
            similarityScores = []
            userRatings = []
            UmovieRating = ratings_matrix[Umovie, :]
            for Smovie in seen:
                # add user's rating for similar movies
                userRatings.append(user_ratings[Smovie])
                # append similarity between desired movie and other seen movies
                SmovieRating = ratings_matrix[Smovie, :]
                similarityScores.append(
                    self.similarity(UmovieRating, SmovieRating))
            unseenEstimateTracker[Umovie] = np.dot(similarityScores, userRatings)
        
        sorted_dict = dict(sorted(unseenEstimateTracker.items(), key=lambda item: item[1], reverse=True))
        recommendations = list(sorted_dict.keys())
        recommendations = recommendations[:k]
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """

if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
