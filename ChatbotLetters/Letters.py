# dboyer2@live.nmhu.edu
"""
Module for generating words/sentences by letter
"""
from Graph import *
import random
import os
import os.path
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


class Dictionary(object):
    """ dict word container holds word counts """
    # constructor initializes words_dict
    def __init__(self):
        self.words_dict = {}

    def add_word(self, word, value):
        self.words_dict.update({word: + value})

    def add_words(self, words):
        for word in words:
            self.words_dict.update({word: + 1})

    def get_dict(self):
        return self.words_dict

    def update_word_count(self, word, value):
        self.words_dict[word] += value


# dictionary to store word values
dictionary = Dictionary()


class Letter(object):
    """ Letter object for word and sentence structures - contains weights and letter"""
    # constructor takes name: string weight: int and graph: array/list
    def __init__(self, name, weight, graph=None):
        self.name = name
        self.weight = weight
        if graph is None:
            graph = []
        self.graph = graph
        self.prev_next_weight = [[0]*99 for i in range(99)]
        self.last_next_weight = [0 for i in range(99)]
        self.test_weights = np.full((99, 99, 99, 99), 0)
        self.previous = None
        self.pre = None

    # TODO edit
    # function to get next letter based upon 4d graph with some randomness
    def get_next_two_letter_4d(self, prev, pre):
        if prev is not None:
            largest1 = 0
            pos = []
            letter_num1 = 0
            letter_num2 = 0
            largest_array1 = []
            letter_num_array1 = []
            letter_num_array2 = []
            for i in range(0, len(self.test_weights)):
                for j in range(0, len(self.test_weights)):
                    for k in range(0, len(self.test_weights)):
                        for h in range(0, len(self.test_weights)):
                            if i == get_number_from_letter(prev.get_name()) and k == get_number_from_letter(
                                    pre.get_name()):
                                if self.test_weights[i][j][k][h] > largest1:
                                    pos.append((i, j, k, h))
                                    largest1 = self.test_weights[i][j][k][h]
                                    letter_num1 = j
                                    letter_num2 = h
                                    largest_array1.append(largest1)
                                    letter_num_array1.append(letter_num1)
                                    letter_num_array2.append(letter_num2)

            if self.graph[letter_num1].get_name() != " ":
                if largest1 == 0:
                    self.graph[77].set_previous(prev)
                    self.graph[77].set_pre(prev.get_prev_letter())
                    second_letter = generate_random_letter(self.graph)
                    second_letter.set_previous(self.graph[77])
                    second_letter.set_pre(self.graph[77].get_prev_letter())
                    return self.graph[77], second_letter
                elif len(largest_array1) == 1:
                    print("The only letter is: " + get_letter_from_number(letter_num_array1[0]))
                # sorting array of largest next letters
                for i in range(len(largest_array1)):
                    for j in range(i + 1, len(largest_array1)):

                        if largest_array1[i] > largest_array1[j]:
                            largest_array1[i], largest_array1[j] = largest_array1[j], largest_array1[i]
                            letter_num_array1[i], letter_num_array1[j] = letter_num_array1[j], letter_num_array1[i]
                # getting random probability from length of largest letter array
                if len(letter_num_array1) > 1:
                    prob_array = []
                    largest_sum = 0
                    for i in range(0, len(letter_num_array1)):
                        largest_sum += largest_array1[i]

                    print("This is the largest sum " + str(largest_sum) + "\n")
                    for i in range(0, len(largest_array1)):
                        if i > 0:
                            prob_array.append(
                                (round(largest_array1[i] / largest_sum, 5) * 100000) + prob_array[i - 1])
                            print(
                                str(prob_array[i]) + " letter is : " + get_letter_from_number(letter_num_array1[i]))
                        else:
                            prob_array.append((round(largest_array1[i] / largest_sum, 5) * 100000))
                            print(
                                str(prob_array[i]) + " letter is : " + get_letter_from_number(letter_num_array1[i]))
                        if prob_array[i] >= 50000:
                            temp = prob_array[i] - 50000
                            prob_array[i] += temp
                            prob_array[i] -= (int(temp / (len(prob_array))))
                            for j in range(0, len(prob_array) - 1):
                                prob_array[j] += int(temp / (len(prob_array)))
                            print("\nRedistributed probabilities\n")

                    rand_float = random.random() * 100000
                    print("Random number is : " + str(rand_float) + "\n")
                    for i in range(0, len(prob_array)):
                        if i == 0:
                            if 0 <= rand_float <= int(prob_array[i]):
                                letter_num1 = letter_num_array1[i]
                                letter_num2 = letter_num_array2[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array1[i]))
                                print("\n Second chosen letter is : " + get_letter_from_number(letter_num_array2[i]))

                        elif len(prob_array) >= 2 and 1 <= i < len(prob_array) - 1:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num1 = letter_num_array1[i]
                                letter_num2 = letter_num_array2[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array1[i]))
                                print("\n Second chosen letter is : " + get_letter_from_number(letter_num_array2[i]))

                        elif i == len(prob_array) - 1:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num1 = letter_num_array1[i]
                                letter_num2 = letter_num_array2[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array1[i]))
                                print("\n Second chosen letter is : " + get_letter_from_number(letter_num_array2[i]))

                        else:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num1 = letter_num_array1[i]
                                letter_num2 = letter_num_array2[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array1[i]))
                                print("\n Second chosen letter is : " + get_letter_from_number(letter_num_array2[i]))

            if prev.get_name() == "." or prev.get_name() == "!" or prev.get_name() == "?":
                self.graph[77].set_previous(prev)
                self.graph[77].set_pre(prev.get_prev_letter())
                letter = generate_random_letter(self.graph)
                letter.set_previous(self.graph[77])
                letter.set_pre(self.graph[77].get_prev_letter())
                return self.graph[77], letter
            fletter = None
            sletter = None
            for i in range(0, len(self.graph)):
                if get_letter_from_number(letter_num1) == self.graph[i].get_name():
                    self.graph[i].set_previous(self.graph[get_number_from_letter(self.get_name())])
                    self.graph[i].set_pre(self.get_prev_letter())
                    fletter = self.graph[i]
            for i in range(0, len(self.graph)):
                if get_letter_from_number(letter_num2) == self.graph[i].get_name():
                    self.graph[i].set_previous(fletter)
                    self.graph[i].set_pre(fletter.get_prev_letter())
                    sletter = self.graph[i]
                    return fletter, sletter
            if self.graph[get_number_from_letter(get_letter_from_number(letter_num1))] not in self.graph:
                self.graph[77].set_previous(self.graph[get_number_from_letter(self.get_name())])
                self.graph[77].set_pre(self.get_prev_letter())
                letter = generate_random_letter(self.graph)
                letter.set_previous(self.graph[77])
                letter.set_pre(self.graph[77].get_prev_letter())
                return self.graph[77], letter
        else:
            return self.graph[77], self.graph[77]

    # TODO test - working but VERY SLOW
    # function to get next letter based upon 4d graph with some randomness
    def get_next_letter_4d(self, prev, pre):
        if prev is not None:
            largest = 0
            letter_num = 0
            largest_array = []
            letter_num_array = []
            for i in range(0, len(self.test_weights)):
                for j in range(0, len(self.test_weights)):
                    for k in range(0, len(self.test_weights)):
                        for h in range(0, len(self.test_weights)):
                            if i == get_number_from_letter(prev.get_name()) and k == get_number_from_letter(pre.get_name()):
                                if self.test_weights[i][j][k][h] > largest:
                                    largest = self.test_weights[i][j][k][h]
                                    letter_num = j
                                    largest_array.append(largest)
                                    letter_num_array.append(letter_num)
                                    
            if self.graph[letter_num].get_name() != " ":
                if largest == 0:
                    self.graph[77].set_previous(prev)
                    self.graph[77].set_pre(prev.get_prev_letter())
                    return self.graph[77]
                elif len(largest_array) == 1:
                    print("The only letter is: " + get_letter_from_number(letter_num_array[0]))
                # sorting array of largest next letters
                for i in range(len(largest_array)):
                    for j in range(i + 1, len(largest_array)):

                        if largest_array[i] > largest_array[j]:
                            largest_array[i], largest_array[j] = largest_array[j], largest_array[i]
                            letter_num_array[i], letter_num_array[j] = letter_num_array[j], letter_num_array[i]
                # getting random probability from length of largest letter array
                if len(letter_num_array) > 1:
                    prob_array = []
                    largest_sum = 0
                    for i in range(0, len(letter_num_array)):
                        largest_sum += largest_array[i]

                    print("This is the largest sum " + str(largest_sum) + "\n")
                    for i in range(0, len(largest_array)):
                        if i > 0:
                            prob_array.append((round(largest_array[i] / largest_sum, 5) * 100000) + prob_array[i - 1])
                            print(str(prob_array[i]) + " letter is : " + get_letter_from_number(letter_num_array[i]))
                        else:
                            prob_array.append((round(largest_array[i] / largest_sum, 5) * 100000))
                            print(str(prob_array[i]) + " letter is : " + get_letter_from_number(letter_num_array[i]))
                        # TODO not changing var
                        if prob_array[i] >= 50000:
                            temp = prob_array[i] - 50000
                            prob_array[i] += temp
                            prob_array[i] -= (int(temp/(len(prob_array))))
                            for j in range(0, len(prob_array)-1):
                                prob_array[j] += int(temp/(len(prob_array)))
                            print("\nRedistributed probabilities\n")

                    rand_float = random.random() * 100000
                    print("Random number is : " + str(rand_float) + "\n")
                    for i in range(0, len(prob_array)):
                        if i == 0:
                            if 0 <= rand_float <= int(prob_array[i]):
                                letter_num = letter_num_array[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array[i]))
                        elif len(prob_array) >= 2 and 1 <= i < len(prob_array) - 1:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num = letter_num_array[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array[i]))
                        elif i == len(prob_array) - 1:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num = letter_num_array[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array[i]))
                        else:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num = letter_num_array[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array[i]))

            if prev.get_name() == "." or prev.get_name() == "!" or prev.get_name() == "?":
                self.graph[77].set_previous(prev)
                self.graph[77].set_pre(prev.get_prev_letter())
                return self.graph[77]
            for i in range(0, len(self.graph)):
                if get_letter_from_number(letter_num) == self.graph[i].get_name():
                    self.graph[i].set_previous(self.graph[get_number_from_letter(self.get_name())])
                    self.graph[i].set_pre(self.get_prev_letter())
                    return self.graph[i]
            if self.graph[get_number_from_letter(get_letter_from_number(letter_num))] not in self.graph:
                self.graph[77].set_previous(self.graph[get_number_from_letter(self.get_name())])
                self.graph[77].set_pre(self.get_prev_letter())
                return self.graph[77]
        else:
            return self.graph[77]

    # function to find path to period - while loops don't work for some reason
    def get_next_sentence_path(self, prev, current_index):
        if prev is not None:
            largest = 0
            total_weight = 0
            letter_num = 0
            i, j = 0, 0
            sentence_path = [self.graph[current_index]]
            while i < len(self.graph[current_index].prev_next_weight):
                print(i)
                while j < len(self.graph[current_index].prev_next_weight):
                    print(j)
                    if i == get_number_from_letter(prev.get_name()):
                        print("PRINTING " + prev.get_name())
                        if self.graph[current_index].prev_next_weight[i][j] > largest:
                            largest = self.graph[current_index].prev_next_weight[i][j]
                            total_weight += largest
                            letter_num = j
                            if j == len(self.prev_next_weight)-1 and self.graph[letter_num].get_name() != '.' and \
                                    self.graph[letter_num].get_name() != '!' and self.graph[letter_num].get_name() != '?':
                                if largest == 0:
                                    letter_num = random.randint(10, 61)
                                sentence_path.append(self.graph[letter_num])
                                self.graph[letter_num].set_previous(prev)
                                current_index = letter_num
                                prev = self.graph[letter_num]
                                i = j
                                j = 0
                            elif self.graph[letter_num].get_name() == '.' or self.graph[letter_num].get_name() == '!' \
                                    or self.graph[letter_num].get_name() == '?':
                                sentence_path.append(self.graph[letter_num])
                                return sentence_path, total_weight
                    j += 1
                i += 1

            return sentence_path, total_weight

    # function to return predictive letter based on weights with no randomness
    def get_next_letter_no_random(self, prev, first=False):
        if prev is not None:
            # for i is in range of previous letter and j in range of next
            # if i matches previous number representation of letter
            # it tries to find the largest weighted letter next which is j
            largest = 0
            letter_num = 0
            largest_array = []
            letter_num_array = []
            for i in range(0, len(self.prev_next_weight)):
                for j in range(0, len(self.prev_next_weight)):
                    if i == get_number_from_letter(prev.get_name()):
                        if self.prev_next_weight[i][j] > largest:
                            largest = self.prev_next_weight[i][j]
                            letter_num = j
                            largest_array.append(largest)
                            letter_num_array.append(letter_num)

            if prev.get_name() == "." or prev.get_name() == "!" or prev.get_name() == "?":
                self.graph[77].set_previous(self.graph[62])
                return self.graph[77]
            for i in range(0, len(self.graph)):
                if get_letter_from_number(letter_num) == self.graph[i].get_name():
                    self.graph[i].set_previous(self.graph[get_number_from_letter(self.get_name())])
                    return self.graph[i]
            if self.graph[get_number_from_letter(get_letter_from_number(letter_num))] not in self.graph:
                self.graph[77].set_previous(self.graph[get_number_from_letter(self.get_name())])
                return self.graph[77]
            print("PRINTING returning .")

    # TODO add duplicate checking with lcs or substring
    # returns the next predictive letter based on weights
    def get_next_letter_no_dup(self, prev, first=False, words=[]):
        if prev is not None:
            # for i is in range of previous letter and j in range of next
            # if i matches previous number representation of letter
            # it tries to find the largest weighted letters next which is j
            largest = 0
            letter_num = 0
            largest_array = []
            letter_num_array = []
            for i in range(0, len(self.prev_next_weight)):
                for j in range(0, len(self.prev_next_weight)):
                    if i == get_number_from_letter(prev.get_name()):
                        if self.prev_next_weight[i][j] > largest:
                            largest = self.prev_next_weight[i][j]
                            letter_num = j
                            largest_array.append(largest)
                            letter_num_array.append(letter_num)
                            # print(str(largest))

            if self.graph[letter_num].get_name() != " ":
                # sorting array of largest next letters
                for i in range(len(largest_array)):
                    for j in range(i + 1, len(largest_array)):

                        if largest_array[i] > largest_array[j]:
                            largest_array[i], largest_array[j] = largest_array[j], largest_array[i]
                            letter_num_array[i], letter_num_array[j] = letter_num_array[j], letter_num_array[i]
                # getting random probability from length of largest letter array
                if len(letter_num_array) > 1:
                    prob_array = []
                    largest_sum = 0
                    for i in range(0, len(letter_num_array)):
                        largest_sum += largest_array[i]
                    for i in range(0, len(largest_array)):
                        prob_array.append(round(largest_array[i] / largest_sum, 3) * 1000)
                    rand_float = random.random() * 1000
                    for i in range(0, len(prob_array)):
                        if i == 0:
                            if rand_float in range(0, int(prob_array[i])):
                                letter_num = letter_num_array[i]
                        else:
                            if rand_float in range(int(prob_array[i - 1]), int(prob_array[i])) or \
                                    rand_float in range(int(prob_array[i], 1000)):
                                letter_num = letter_num_array[i]

            if prev.get_name() == "." or prev.get_name() == "!" or prev.get_name() == "?":
                self.graph[77].set_previous(self.graph[62])
                return self.graph[77]
            for i in range(0, len(self.graph)):
                if get_letter_from_number(letter_num) == self.graph[i].get_name():
                    self.graph[i].set_previous(self.graph[get_number_from_letter(self.get_name())])
                    return self.graph[i]
            if self.graph[get_number_from_letter(get_letter_from_number(letter_num))] not in self.graph:
                self.graph[77].set_previous(self.graph[get_number_from_letter(self.get_name())])
                return self.graph[77]
            print("PRINTING returning .")

    # returns the next predictive letter based on weights
    def get_next_letter(self, prev, first=False):
        if prev is not None:
            # for i is in range of previous letter and j in range of next
            # if i matches previous number representation of letter
            # it tries to find the largest weighted letters next which is j
            largest = 0
            letter_num = 0
            largest_array = []
            letter_num_array = []
            for i in range(0, len(self.prev_next_weight)):
                for j in range(0, len(self.prev_next_weight)):
                    if i == get_number_from_letter(prev.get_name()):
                        if self.prev_next_weight[i][j] > largest:
                            largest = self.prev_next_weight[i][j]
                            letter_num = j
                            largest_array.append(largest)
                            letter_num_array.append(letter_num)
                            #print(str(largest))

            if self.graph[letter_num].get_name() != " ":
                if largest == 0:
                    self.graph[77].set_previous(prev)
                    self.graph[77].set_pre(prev.get_prev_letter())
                    return self.graph[77]
                elif len(largest_array) == 1:
                    print("The only letter is: " + get_letter_from_number(letter_num_array[0]))
                # sorting array of largest next letters
                for i in range(len(largest_array)):
                    for j in range(i + 1, len(largest_array)):

                        if largest_array[i] > largest_array[j]:
                            largest_array[i], largest_array[j] = largest_array[j], largest_array[i]
                            letter_num_array[i], letter_num_array[j] = letter_num_array[j], letter_num_array[i]
                # getting random probability from length of largest letter array
                if len(letter_num_array) > 1:
                    prob_array = []
                    largest_sum = 0
                    for i in range(0, len(letter_num_array)):
                        largest_sum += largest_array[i]

                    print("This is the largest sum " + str(largest_sum) + "\n")
                    for i in range(0, len(largest_array)):
                        if i > 0:
                            prob_array.append((round(largest_array[i] / largest_sum, 5) * 100000) + prob_array[i - 1])
                            print(str(prob_array[i]) + " letter is : " + get_letter_from_number(letter_num_array[i]))
                        else:
                            prob_array.append((round(largest_array[i] / largest_sum, 5) * 100000))
                            print(str(prob_array[i]) + " letter is : " + get_letter_from_number(letter_num_array[i]))
                        if prob_array[i] >= 50000:
                            temp = prob_array[i] - 50000
                            prob_array[i] += temp
                            prob_array[i] -= (int(temp / (len(prob_array))))
                            for j in range(0, len(prob_array) - 1):
                                prob_array[j] += int(temp / (len(prob_array)))
                            print("\nRedistributed probabilities\n")

                    rand_float = random.random() * 100000
                    print("Random number is : " + str(rand_float) + "\n")
                    for i in range(0, len(prob_array)):
                        if i == 0:
                            if 0 <= rand_float <= int(prob_array[i]):
                                letter_num = letter_num_array[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array[i]))
                        elif len(prob_array) >= 2 and 1 <= i < len(prob_array) - 1:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num = letter_num_array[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array[i]))
                        elif i == len(prob_array) - 1:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num = letter_num_array[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array[i]))
                        else:
                            if int(prob_array[i - 1]) <= rand_float <= int(prob_array[i]):
                                letter_num = letter_num_array[i]
                                print(str(prob_array[i]) + " chosen letter is : "
                                      + get_letter_from_number(letter_num_array[i]))

            if prev.get_name() == "." or prev.get_name() == "!" or prev.get_name() == "?":
                self.graph[77].set_previous(self.graph[62])
                return self.graph[77]
            for i in range(0, len(self.graph)):
                if get_letter_from_number(letter_num) == self.graph[i].get_name():
                    self.graph[i].set_previous(self.graph[get_number_from_letter(self.get_name())])
                    return self.graph[i]
            if self.graph[get_number_from_letter(get_letter_from_number(letter_num))] not in self.graph:
                self.graph[77].set_previous(self.graph[get_number_from_letter(self.get_name())])
                return self.graph[77]
            print("PRINTING returning .")

    # adds weight to letter
    def add_weight_first(self, next, weight):
        letter_num = get_number_from_letter(next.get_name())
        self.first_next_weight[letter_num] += weight

    # adds weight to letter with previous and/or next letter weights
    def add_weight(self, next=None, prev=None, weight=0):
        if next is not None and prev is not None:
            next_letter_num = get_number_from_letter(next.get_name())
            prev_letter_num = get_number_from_letter(prev.get_name())
            self.prev_next_weight[prev_letter_num][next_letter_num] += weight
            #print("PRINTING added weight: ", self.prev_next_weight[prev_letter_num][next_letter_num])
        else:
            prev_letter_num = get_number_from_letter(next.get_name())
            self.last_next_weight[prev_letter_num] += weight

    # adds weight to letter with two previous and/or next letter weights
    def add_weight_4d(self, next=None, ne=None, prev=None, pre=None, weight=0):
        if next is not None and prev is not None and ne is not None and pre is not None:
            next_letter_num = get_number_from_letter(next.get_name())
            ne_letter_num = get_number_from_letter(ne.get_name())
            prev_letter_num = get_number_from_letter(prev.get_name())
            pre_letter_num = get_number_from_letter(pre.get_name())

            self.test_weights[prev_letter_num][next_letter_num][pre_letter_num][ne_letter_num] += weight
        else:
            prev_letter_num = get_number_from_letter(next.get_name())
            self.last_next_weight[prev_letter_num] += weight

    # returns previous letter based upon weights
    def get_prev_letter(self, first=False):
        return self.previous

    # getters
    def get_name(self):
        return self.name

    def get_weight(self):
        return self.weight

    def get_graph(self):
        return self.graph

    def get_pre(self):
        return self.pre

    # setters
    def set_name(self, name):
        self.name = name

    def set_previous(self, letter):
        self.previous = letter

    def set_pre(self, letter):
        self.pre = letter

    def set_weight(self, weight):
        self.weight = weight

    def set_graph(self, graph):
        self.graph = graph

    # function to print all weights
    def print_weights(self, index=None):
        if index is not None:
            # for i, w in enumerate(self.prev_next_weight):
            print("PRINTING prev/next weights at index " + str(index) + " " + str(self.prev_next_weight[index]))
        else:
            for i, w in enumerate(self.prev_next_weight):
                print("PRINTING prev/next weights at index " + str(i) + " " + str(self.prev_next_weight[i]))
                # print("PRINTING weights at index " + str(i) + " " + str(self.first_next_weight[i]))

    # function to check if all weights are zeroed out
    def check_weights_zero(self):
        num = 0
        for i in range(0, len(self.prev_next_weight)):
            for j in range(0, len(self.prev_next_weight)):
                if self.prev_next_weight[i][j] > num:
                    num = self.prev_next_weight[i][j]
        if num == 0:
            return True
        else:
            return False
# end class Letter


class Word(object):
    """ Word object for sentence structures - contains array of letter objects"""
    # constructor takes name: string and letters: array/list
    def __init__(self, name=None, letters=None):
        if name is None:
            name = ""
        self.name = name
        if letters is None:
            letters = []
            self.count = 0
        else:
            self.count = len(letters)
        self.letters = letters

    # getters
    def get_name(self):
        return self.name

    def get_count(self):
        return self.count

    def get_letters(self):
        return self.letters

    def get_letter(self, pos):
        return self.letters[pos]

    # setters
    def set_name(self, name):
        self.name = name

    def set_count(self, count):
        self.count = count

    def set_letters(self, letters=[]):
        self.letters = letters
# end class Word


class Sentence(object):
    """ Sentence object parses strings/lists into word and letter objects - contains array of word objects"""
    # constructor takes name: string words: array/list and graph: array/list
    def __init__(self, name=None, words=None, graph=None):
        if graph is None:
            graph = []
        self.graph = graph
        self.count = 0
        self.words = words

        if words is None:
            self.words = []

        if name is None:
            name = ""
            self.name = name
        elif type(name) == list:
            self.parse_sentence_array(name)
        else:
            self.parse_sentence_string(name)

        if words:
            if words is not None:
                self.parse_sentence_array(words)
            else:
                self.parse_sentence_string(name)

    # parses sentence as string
    def parse_sentence_string(self, words):
        count = 0
        sentence_array = []
        new_word_letters = []
        sentence_string = ""
        sentence_word = ""

        prev_word = ""
        for index, word in enumerate(words):
            for i in range(-1, len(self.graph)):
                if words[index] == self.graph[i].get_name():
                    new_word_letters.append(self.graph[i])
                if i <= 61 and word == self.graph[i].get_name() and index < len(words)-1:
                    sentence_word += self.graph[i].get_name()
                    if not words[index+1].isnumeric() and not words[index+1].isalpha() and words[index+1] != "'" \
                            and words[index+1] != "-":
                        sentence_string += sentence_word
                        new_word_letters.append(self.graph[i])
                        new_word = Word(sentence_word, new_word_letters)
                        sentence_array.append(new_word)
                        sentence_word = ""
                        new_word_letters = []
                        count += 1
                    else:
                        new_word_letters.append(self.graph[i])
                elif i > 61 and word == self.graph[i].get_name():
                    if word != "'" and word != "-":
                        sentence_word = self.graph[i].get_name()
                        sentence_string += sentence_word
                        new_word_letters.append(self.graph[i])
                        new_word = Word(sentence_word, new_word_letters)
                        sentence_array.append(new_word)
                        sentence_word = ""
                        new_word_letters = []
                        count += 1
                    else:
                        if index == len(words)-1 or words[index+1] == " ":
                            sentence_word += self.graph[i].get_name()
                            sentence_string += sentence_word
                            new_word_letters.append(self.graph[i])
                            new_word = Word(sentence_word, new_word_letters)
                            sentence_array.append(new_word)
                            sentence_word = ""
                            new_word_letters = []
                            count += 1
                        else:
                            sentence_word += self.graph[i].get_name()
                            new_word_letters.append(self.graph[i])

        self.set_name(sentence_string)
        self.set_words(sentence_array)
        self.set_count(count)

    # parses sentence as list/array
    def parse_sentence_array(self, words):
        count = 0
        sentence_array = []
        sentence_string = ""
        for word in words:
            count += 1
            if count == len(words):
                sentence_string += word
            elif count == 1:
                for i in range(0, len(word)):
                    if i == 0:
                        sentence_string += word[i].upper()
                    elif i == len(word)-1:
                        sentence_string += word[i] + " "
                    else:
                        sentence_string += word[i]
            else:
                sentence_string += word + " "
            new_word_letters = []
            for letter in word:
                for let in self.graph:
                    if letter == let:
                        new_word_letters.append(let)
            new_word = Word(word, new_word_letters)
            sentence_array.append(new_word)
        self.set_name(sentence_string)
        self.set_words(sentence_array)
        self.set_count(count)

    # getters
    def get_name(self):
        return self.name

    def get_count(self):
        return self.count

    def get_words(self):
        return self.words

    def get_word(self, pos):
        return self.words[pos]

    # setters
    def set_name(self, name):
        self.name = name

    def set_count(self, count):
        self.count = count

    def set_words(self, words):
        self.words = words
# end class Sentence


# returns a letter char/string based upon number
def get_letter_from_number(num):
    match num:
        case 0:
            return '0'
        case 1:
            return '1'
        case 2:
            return '2'
        case 3:
            return '3'
        case 4:
            return '4'
        case 5:
            return '5'
        case 6:
            return '6'
        case 7:
            return '7'
        case 8:
            return '8'
        case 9:
            return '9'
        case 10:
            return 'a'
        case 11:
            return 'b'
        case 12:
            return 'c'
        case 13:
            return 'd'
        case 14:
            return 'e'
        case 15:
            return 'f'
        case 16:
            return 'g'
        case 17:
            return 'h'
        case 18:
            return 'i'
        case 19:
            return 'j'
        case 20:
            return 'k'
        case 21:
            return 'l'
        case 22:
            return 'm'
        case 23:
            return 'n'
        case 24:
            return 'o'
        case 25:
            return 'p'
        case 26:
            return 'q'
        case 27:
            return 'r'
        case 28:
            return 's'
        case 29:
            return 't'
        case 30:
            return 'u'
        case 31:
            return 'v'
        case 32:
            return 'w'
        case 33:
            return 'x'
        case 34:
            return 'y'
        case 35:
            return 'z'
        case 36:
            return 'A'
        case 37:
            return 'B'
        case 38:
            return 'C'
        case 39:
            return 'D'
        case 40:
            return 'E'
        case 41:
            return 'F'
        case 42:
            return 'G'
        case 43:
            return 'H'
        case 44:
            return 'I'
        case 45:
            return 'J'
        case 46:
            return 'K'
        case 47:
            return 'L'
        case 48:
            return 'M'
        case 49:
            return 'N'
        case 50:
            return 'O'
        case 51:
            return 'P'
        case 52:
            return 'Q'
        case 53:
            return 'R'
        case 54:
            return 'S'
        case 55:
            return 'T'
        case 56:
            return 'U'
        case 57:
            return 'V'
        case 58:
            return 'W'
        case 59:
            return 'X'
        case 60:
            return 'Y'
        case 61:
            return 'Z'
        case 62:
            return '.'
        case 63:
            return '?'
        case 64:
            return '!'
        case 65:
            return ','
        case 66:
            return ':'
        case 67:
            return ';'
        case 68:
            return '/'
        case 69:
            return '-'
        case 70:
            return '['
        case 71:
            return ']'
        case 72:
            return '{'
        case 73:
            return '}'
        case 74:
            return '('
        case 75:
            return ')'
        case 76:
            return "'"
        case 77:
            return " "
        case 78:
            return "~"
        case 79:
            return "@"
        case 80:
            return "#"
        case 81:
            return "$"
        case 82:
            return "%"
        case 83:
            return "^"
        case 84:
            return "&"
        case 85:
            return "*"
        case 86:
            return "_"
        case 87:
            return "+"
        case 88:
            return "="
        case 89:
            return "|"
        case 90:
            return "<"
        case 91:
            return ">"
        case 92:
            return "`"
        case 93:
            return '"'
        case 94:
            return '\\'
        case 95:
            return '—'
        case 96:
            return '\n'
        case 97:
            return '’'
        case 98:
            return '“'
        case _:
            return "\n"
# end get_letter_from_number(num):


# returns a number from given letter to match array
def get_number_from_letter(letter):
    match letter:
        case '0':
            return 0
        case '1':
            return 1
        case '2':
            return 2
        case '3':
            return 3
        case '4':
            return 4
        case '5':
            return 5
        case '6':
            return 6
        case '7':
            return 7
        case '8':
            return 8
        case '9':
            return 9
        case 'a':
            return 10
        case 'b':
            return 11
        case 'c':
            return 12
        case 'd':
            return 13
        case 'e':
            return 14
        case 'f':
            return 15
        case 'g':
            return 16
        case 'h':
            return 17
        case 'i':
            return 18
        case 'j':
            return 19
        case 'k':
            return 20
        case 'l':
            return 21
        case 'm':
            return 22
        case 'n':
            return 23
        case 'o':
            return 24
        case 'p':
            return 25
        case 'q':
            return 26
        case 'r':
            return 27
        case 's':
            return 28
        case 't':
            return 29
        case 'u':
            return 30
        case 'v':
            return 31
        case 'w':
            return 32
        case 'x':
            return 33
        case 'y':
            return 34
        case 'z':
            return 35
        case 'A':
            return 36
        case 'B':
            return 37
        case 'C':
            return 38
        case 'D':
            return 39
        case 'E':
            return 40
        case 'F':
            return 41
        case 'G':
            return 42
        case 'H':
            return 43
        case 'I':
            return 44
        case 'J':
            return 45
        case 'K':
            return 46
        case 'L':
            return 47
        case 'M':
            return 48
        case 'N':
            return 49
        case 'O':
            return 50
        case 'P':
            return 51
        case 'Q':
            return 52
        case 'R':
            return 53
        case 'S':
            return 54
        case 'T':
            return 55
        case 'U':
            return 56
        case 'V':
            return 57
        case 'W':
            return 58
        case 'X':
            return 59
        case 'Y':
            return 60
        case 'Z':
            return 61
        case '.':
            return 62
        case '?':
            return 63
        case '!':
            return 64
        case ',':
            return 65
        case ':':
            return 66
        case ';':
            return 67
        case '/':
            return 68
        case "-":
            return 69
        case '[':
            return 70
        case ']':
            return 71
        case '{':
            return 72
        case '}':
            return 73
        case '(':
            return 74
        case ')':
            return 75
        case "'":
            return 76
        case ' ':
            return 77
        case '~':
            return 78
        case '@':
            return 79
        case '#':
            return 80
        case '$':
            return 81
        case '%':
            return 82
        case '^':
            return 83
        case '&':
            return 84
        case '*':
            return 85
        case '_':
            return 86
        case '+':
            return 87
        case '=':
            return 88
        case '|':
            return 89
        case '<':
            return 90
        case '>':
            return 91
        case "`":
            return 92
        case '"':
            return 93
        case '\\':
            return 94
        case '—':
            return 95
        case '\n':
            return 96
        case '’':
            return 97
        case '“':
            return 98
        case _:
            return 96


# initializes an array of letters for graph usage in letters/words/sentences
def initialize_letter_array():
    letters_array = []
    for letter in [get_letter_from_number(i) for i in range(0, 99)]:
        letters_array.append(Letter(letter, 0, letters_array))
    return letters_array
# end initialize_letter_array()


# generates first letter based on graph
def generate_first_letter(graph):
    index = random.randint(36, 61)
    return graph[index]
# end generate_letter()


# function to generate random letter
def generate_random_letter(graph):
    index = random.randint(10, 36)
    return graph[index]


# generates a paragraph
def generate_paragraph(graph):
    first_letter = generate_first_letter(graph)
    first_letter.set_previous(graph[77])
    if first_letter.check_weights_zero():
        print("Zeroed")
    count = 0
    sentence_array = []
    sentence_array.append(generate_sentence_by_letter(first_letter, "", graph))
    while count < 5:
        letter = generate_first_letter(graph)
        letter.set_previous(graph[77])
        sentence_array.append(generate_sentence_by_letter(letter, "", graph))
        # REMOVED
        # graph[get_number_from_letter(sentence_array[count][len(sentence_array
        #                                                                                                      [count])-1])]
        count += 1
    return sentence_array


# function to generate sentence using get_next_sentence_path - non-functional
def generate_sentence(graph):
    letter = generate_first_letter(graph)
    letter.set_previous(graph[77])
    #sentence_array, total_weight = letter.get_next_sentence_path(graph[77], get_number_from_letter(letter.get_name()))
    #for letter in sentence_array:
    #    sentence += letter.get_name()
    return generate_sentence_by_letter(letter, "", graph)


# function to generate 4d sentence
def generate_4d_sentence(graph):
    f_letter = generate_first_letter(graph)
    f_letter.set_previous(graph[77])
    f_letter.set_pre(graph[62])
    return generate_sentence_by_letter_4d(f_letter, "", graph)


# function to generate 4d sentence
def generate_4d_sentence_2letter(graph):
    f_letter = generate_first_letter(graph)
    f_letter.set_previous(graph[77])
    f_letter.set_pre(graph[62])
    return generate_sentence_by_2letter_4d(f_letter, "", graph)


# function to generate sentence by word
def generate_sentence_by_word(num_words, graph, punctuation):
    count = 0
    sentence = ""

    letter = generate_first_letter(graph)
    letter.set_previous(graph[77])
    sentence += generate_word_by_letter(letter, "", graph)

    while count <= num_words - 1:
        letter = generate_random_letter(graph)
        letter.set_previous(graph[77])
        sentence += " " + generate_word_by_letter(letter, "", graph)
        count += 1
    sentence += punctuation
    return sentence


# function to generate sentence by word no random
def generate_non_random_sentence_by_word(num_words, graph, punctuation):
    count = 0
    sentence = ""

    letter = generate_first_letter(graph)
    letter.set_previous(graph[77])
    sentence += generate_non_random_word_by_letter(letter, "", graph)

    while count <= num_words - 1:
        letter = generate_random_letter(graph)
        letter.set_previous(graph[77])
        sentence += " " + generate_non_random_word_by_letter(letter, "", graph)
        count += 1
    sentence += punctuation
    return sentence


# function to chat with user
def generate_chat(num_words, graph, punctuation, prev):
    count = 0
    sentence = ""

    rand = random.randint(62, 64)
    punctuation = graph[rand].get_name()
    letter = graph[77]
    letter.set_previous(prev)
    next_letter = letter.get_next_letter(prev)
    sentence += generate_non_random_word_by_letter(next_letter, "", graph)

    while count <= num_words - 1:
        letter = generate_random_letter(graph)
        letter.set_previous(graph[77])
        sentence += " " + generate_non_random_word_by_letter(letter, "", graph)
        count += 1
    sentence += punctuation
    return sentence


# function to generate word letter by letter non random
def generate_non_random_word_by_letter(letter, word, graph, count=0):
    word = word
    run = True
    while run:
        if count > 12:
            letter = graph[77]
        next_letter = letter.get_next_letter_no_random(letter.get_prev_letter(), False)
        word += letter.get_name()
        word.strip()
        #print(word)
        while next_letter.get_name() == letter.get_name() and letter.get_name() == letter.get_prev_letter().get_name():
            next_letter = generate_random_letter(graph)
            next_letter.set_previous(letter)
        if next_letter.get_name() == '.' or next_letter.get_name() == '!' or next_letter.get_name() == '?' \
                or next_letter.get_name() == ' ':
            return word
        return generate_non_random_word_by_letter(next_letter, word, graph, count)
    return word


# function to generate word letter by letter
def generate_word_by_letter(letter, word, graph):
    word = word
    run = True
    while run:
        next_letter = letter.get_next_letter(letter.get_prev_letter(), False)
        word += letter.get_name()
        #print(word)
        while next_letter.get_name() == letter.get_name() and letter.get_name() == letter.get_prev_letter().get_name():
            next_letter = generate_random_letter(graph)
            next_letter.set_previous(letter)
        if next_letter.get_name() == '.' or next_letter.get_name() == '!' or next_letter.get_name() == '?'\
                or next_letter.get_name() == ' ':
            return word
        return generate_word_by_letter(next_letter, word, graph)
    return word


# generates next letter based on weighted graph
def generate_sentence_by_letter(letter, sentence, graph, count=0):
    sentence = sentence
    run = True
    while run:
        count += 1
        if count > 90:
            temp = letter
            letter = graph[62]
            letter.set_previous(temp)

        next_letter = letter.get_next_letter(letter.get_prev_letter(), False)
        sentence += letter.get_name()
        print(sentence)
        while next_letter.get_name() == letter.get_name() and letter.get_name() == letter.get_prev_letter().get_name():
            next_letter = generate_random_letter(graph)
            next_letter.set_previous(letter)
        if letter.get_name() == '.' or letter.get_name() == '!' or letter.get_name() == '?'\
                or letter.get_name() == '\n':
            break
        return generate_sentence_by_letter(next_letter, sentence, graph, count)
    return sentence


# TODO add word substring checking
# generates next letter based on weighted graph
def generate_sentence_by_letter_no_dup(letter, sentence, graph, count=0):
    sentence = sentence
    run = True
    while run:
        count += 1
        if count > 90:
            temp = letter
            letter = graph[62]
            letter.set_previous(temp)

        next_letter = letter.get_next_letter(letter.get_prev_letter(), False)
        sentence += letter.get_name()
        sentence_struc = Sentence(sentence, None, graph)
        words = sentence_struc.get_words()
        # TODO do something with words and given sentence
        #print(sentence)
        while next_letter.get_name() == letter.get_name() and letter.get_name() == letter.get_prev_letter().get_name():
            next_letter = generate_random_letter(graph)
            next_letter.set_previous(letter)
        if letter.get_name() == '.' or letter.get_name() == '!' or letter.get_name() == '?'\
                or letter.get_name() == '\n':
            break
        return generate_sentence_by_letter(next_letter, sentence, graph, count)
    return sentence


# TODO make it faster
# generates next letter based on 4d weighted graph
def generate_sentence_by_letter_4d(letter, sentence, graph, count=0):
    sentence = sentence
    run = True
    while run:
        count += 1
        if count > 90:
            temp = letter
            letter = graph[62]
            letter.set_previous(temp)
            letter.set_pre(temp.get_prev_letter())

        next_letter = letter.get_next_letter_4d(letter.get_prev_letter(), letter.get_pre())
        sentence += letter.get_name()
        print(sentence)
        while next_letter.get_name() == letter.get_name() and letter.get_name() == letter.get_prev_letter().get_name():
            next_letter = generate_random_letter(graph)
            next_letter.set_previous(letter)
            next_letter.set_pre(letter.get_prev_letter())
        if letter.get_name() == '.' or letter.get_name() == '!' or letter.get_name() == '?'\
                or letter.get_name() == '\n':
            break
        return generate_sentence_by_letter_4d(next_letter, sentence, graph, count)
    return sentence


# TODO make it faster
# generates next letter based on 4d weighted graph
def generate_sentence_by_2letter_4d(letter, sentence, graph, count=0):
    sentence = sentence
    run = True
    while run:
        count += 1
        if count > 120:
            temp = letter
            letter = graph[62]
            letter.set_previous(temp)
            letter.set_pre(temp.get_prev_letter())
            break

        next_letter, ne_letter = letter.get_next_two_letter_4d(letter.get_prev_letter(), letter.get_pre())
        sentence += letter.get_name()
        while next_letter.get_name() == letter.get_name() and letter.get_name() == letter.get_prev_letter().get_name() \
                or ne_letter.get_name() == next_letter.get_name() and next_letter.get_name() == letter.get_name():
            next_letter = generate_random_letter(graph)
            rand = random.randint(0, 1)
            if random.randint(0, 1) == 1:
                next_letter = generate_random_letter(graph)
                ne_letter.set_previous(next_letter)
                ne_letter.set_pre(next_letter.get_prev_letter())
            else:
                next_letter = graph[77]
                ne_letter.set_previous(next_letter)
                ne_letter.set_pre(next_letter.get_prev_letter())
            next_letter.set_previous(letter)
            next_letter.set_pre(letter.get_prev_letter())
            ne_letter.set_previous(next_letter)
            ne_letter.set_pre(next_letter.get_prev_letter())
        #if letter.get_name() == '.' or letter.get_name() == '!' or letter.get_name() == '?'\
        #        or letter.get_name() == '\n' or letter.get_prev_letter().get_name() == '.' \
        #        or letter.get_prev_letter().get_name() == '!' or letter.get_prev_letter().get_name() == '?':
        #    break
        sentence += next_letter.get_name()
        print(sentence)
        return generate_sentence_by_2letter_4d(ne_letter, sentence, graph, count)
    return sentence


def generate_code_by_letter(letter, sentence, graph):
    sentence = sentence
    run = True
    count = 0
    letter.set_previous(graph[74])
    while run or count < 100:
        next_letter = letter.get_next_letter(letter.get_prev_letter(), False)
        sentence += letter.get_name()
        print(sentence)
        if letter.get_name() == '.' or letter.get_name() == '!' or letter.get_name() == '?'\
                or letter.get_name() == '\n':
            break
        if letter.get_name() == ')':
            break
        count += 1
        return generate_code_by_letter(next_letter, sentence, graph)
    return sentence


# TODO test escape characters maybe
# generates a graph of letters
def generate_letter_graph(g, graph):
    for i in graph:
        g.add_vertex(i.get_name())
    for i in graph:
        for index in graph:
            g.add_edge(i.get_name(), index.get_name(), 0, True)

# end generate_letter_graph()


# function to check if graph is zeroed out
def check_if_graph_zeroed(graph):
    empty = []
    for g in graph:
        empty.append(g.check_weights_zero())
    for e in empty:
        if not e:
            return e
    return True


# TODO edit for sentence lines
# parses input from text files that are conversational
def read_input_text_convo(filename, graph):
    text = ""
    with open("text-files/" + filename + ".txt", encoding="utf8") as file:
        text = file.readlines()

    sentence_string = ""
    array_of_sentences = []

    prev_word = ""
    text = ' '.join(text)
    for index, word in enumerate(text):
        for letter in word:
            if index == 0 and text[index] is not None:
                let_num_next = get_number_from_letter(text[index + 1])
                let_num_prev = get_number_from_letter(graph[77].get_name())
                graph[get_number_from_letter(text[index])].add_weight(graph[let_num_next],
                                                                      graph[let_num_prev], 1)
            elif index == len(text):
                let = text[index - 1]
                let_num = get_number_from_letter(let)
                graph[get_number_from_letter(text[index])].add_weight(None, graph[let_num], 1)
            elif index != len(text) - 1:
                let_num_next = get_number_from_letter(text[index + 1])
                let_num_prev = get_number_from_letter(text[index - 1])
                graph[get_number_from_letter(text[index])].add_weight(graph[let_num_next],
                                                                      graph[let_num_prev], 1)

            sentence_string += letter
            if letter == '.' or letter == '?' or letter == '!' and text[index + 1] != '.':
                array_of_sentences.append(Sentence(sentence_string, None, graph))
                sentence_string = ""
        sentence_string += " "
    print(len(array_of_sentences))
    return graph


# function to read in and weight user input
def read_user_input(text, graph):
    sentence = text
    for index, word in enumerate(sentence):
        if index == 0 and text[index] is not None:
            let_num_next = get_number_from_letter(text[index + 1])
            let_num_prev = get_number_from_letter(graph[77].get_name())
            graph[get_number_from_letter(text[index])].add_weight(graph[let_num_next],
                                                                  graph[let_num_prev], 1)
        elif index == len(text):
            let = text[index - 1]
            let_num = get_number_from_letter(let)
            graph[get_number_from_letter(text[index])].add_weight(None, graph[let_num], 1)
        elif index != len(text) - 1:
            let_num_next = get_number_from_letter(text[index + 1])
            let_num_prev = get_number_from_letter(text[index - 1])
            graph[get_number_from_letter(text[index])].add_weight(graph[let_num_next],
                                                                  graph[let_num_prev], 1)
    return Sentence(sentence, None, graph)


# function to parse input files into 4d graph weights
def read_input_4d(filename, graph):
    text = ""
    if not type(filename) is str:
        raise TypeError("Only strings are allowed")
    elif not type(graph) is list:
        raise TypeError("Only arrays are allowed")
    with open("text-files/" + filename + ".txt", encoding="utf8") as file:
        text = file.readlines()

    sentence_string = ""
    array_of_sentences = []

    prev_word = ""
    text = ' '.join(text)
    for index, word in enumerate(text):
        if index in range(0, 2) and text[index] is not None:
            if index == 0:
                let_num_next = get_number_from_letter(text[index + 1])
                let_num_ne = get_number_from_letter(text[index + 2])
                let_num_prev = get_number_from_letter(graph[77].get_name())
                let_num_pre = get_number_from_letter(graph[62].get_name())
                graph[get_number_from_letter(text[index])].add_weight_4d(graph[let_num_next], graph[let_num_ne],
                                                                         graph[let_num_prev], graph[let_num_pre], 1)
            elif index == 1:
                let_num_next = get_number_from_letter(text[index + 1])
                let_num_ne = get_number_from_letter(text[index + 2])
                let_num_prev = get_number_from_letter(text[index-1])
                let_num_pre = get_number_from_letter(graph[77].get_name())
                graph[get_number_from_letter(text[index])].add_weight_4d(graph[let_num_next], graph[let_num_ne],
                                                                         graph[let_num_prev], graph[let_num_pre], 1)
        elif index == len(text):

            let_num_next = get_number_from_letter(graph[77].get_name())
            let_num_ne = get_number_from_letter(generate_random_letter(graph).get_name())
            let_num_prev = get_number_from_letter(text[index - 1])
            let_num_pre = get_number_from_letter(text[index - 2])
            graph[get_number_from_letter(text[index])].add_weight_4d(graph[let_num_next], graph[let_num_ne],
                                                                     graph[let_num_prev], graph[let_num_pre], 1)
        elif index not in range(len(text) - 3, len(text)):
            let_num_next = get_number_from_letter(text[index + 1])
            let_num_ne = get_number_from_letter(text[index + 2])
            let_num_prev = get_number_from_letter(text[index - 1])
            let_num_pre = get_number_from_letter(text[index - 2])
            graph[get_number_from_letter(text[index])].add_weight_4d(graph[let_num_next], graph[let_num_ne],
                                                                     graph[let_num_prev], graph[let_num_pre], 1)

            sentence_string += word
            if word == '.' or word == '?' or word == '!' and text[index + 1] != '.':
                array_of_sentences.append(Sentence(sentence_string, None, graph))
                sentence_string = ""
        sentence_string += " "
    print(len(array_of_sentences))
    print("Text File " + filename + " Read and Weighted! \n")
    return graph


# parses input text into letters/words/sentences and adds weights to graph
def read_input_text(filename, graph):
    text = ""
    if not type(filename) is str:
        raise TypeError("Only strings are allowed")
    elif not type(graph) is list:
        raise TypeError("Only arrays are allowed")
    with open("text-files/" + filename + ".txt", encoding="utf8") as file:
        text = file.readlines()

    sentence_string = ""
    array_of_sentences = []

    prev_word = ""
    text = ' '.join(text)
    for index, word in enumerate(text):
        if index == 0 and text[index] is not None:
            let_num_next = get_number_from_letter(text[index + 1])
            let_num_prev = get_number_from_letter(graph[77].get_name())
            graph[get_number_from_letter(text[index])].add_weight(graph[let_num_next],
                                                                  graph[let_num_prev], 1)
        elif index == len(text):
            let = text[index-1]
            let_num = get_number_from_letter(let)
            graph[get_number_from_letter(text[index])].add_weight(None, graph[let_num], 1)
        elif index != len(text)-1:
            let_num_next = get_number_from_letter(text[index + 1])
            let_num_prev = get_number_from_letter(text[index - 1])
            graph[get_number_from_letter(text[index])].add_weight(graph[let_num_next],
                                                                  graph[let_num_prev], 1)

            sentence_string += word
            if word == '.' or word == '?' or word == '!' and text[index+1] != '.':
                array_of_sentences.append(Sentence(sentence_string, None, graph))
                sentence_string = ""
        sentence_string += " "
    print(len(array_of_sentences))
    for sentence in array_of_sentences:
        for word in sentence.get_words():
            if word not in dictionary.get_dict():
                dictionary.add_word(word.get_name(), 1)
            else:
                dictionary.update_word_count(word.get_name(), 1)
    print("Text File " + filename + " Read and Weighted! \n")
    return graph
    # TODO find way to parse into sentences with regex elipses


# function to save graph files
def save_graph(graph, name=None):
    if not type(name) is str:
        raise TypeError("Only strings are allowed")
    elif not type(graph) is list:
        raise TypeError("Only arrays are allowed")
    if os.path.isfile(name + ".txt"):
        with open("graphs/" + name + "_graph.txt", 'w') as file:
            for i in range(0, len(graph)-1):
                for j in range(0, len(graph[i].prev_next_weight)-1):
                    for k in range(0, len(graph[i].prev_next_weight) - 1):
                        file.write(str(graph[i].prev_next_weight[j][k]) + "\n")
    else:
        file = open("graphs/" + name + "_graph.txt", 'w')
        for i in range(0, len(graph) - 1):
            for j in range(0, len(graph[i].prev_next_weight) - 1):
                for k in range(0, len(graph[i].prev_next_weight) - 1):
                    file.write(str(graph[i].prev_next_weight[j][k]) + "\n")
        file.close()

    print("Graph " + name + "_graph.txt saved!")


# function to save 4d graph files. Takes too much space
def save_4d_graph(graph, name=None):
    if not type(name) is str:
        raise TypeError("Only strings are allowed")
    elif not type(graph) is list:
        raise TypeError("Only arrays are allowed")
    if os.path.isfile(name + ".txt"):
        with open(name + "_4d_graph.txt", 'w') as file:
            for i in range(0, len(graph)-1):
                for j in range(0, len(graph[i].test_weights) - 1):
                    for k in range(0, len(graph[i].test_weights) - 1):
                        for l in range(0, len(graph[i].test_weights) - 1):
                            for m in range(0, len(graph[i].test_weights) - 1):
                                file.write(str(graph[i].test_weights[j][k][l][m]) + "\n")
    else:
        file = open(name + "_4d_graph.txt", 'w')
        for i in range(0, len(graph) - 1):
            for j in range(0, len(graph[i].test_weights) - 1):
                for k in range(0, len(graph[i].test_weights) - 1):
                    for l in range(0, len(graph[i].test_weights) - 1):
                        for m in range(0, len(graph[i].test_weights) - 1):
                            file.write(str(graph[i].test_weights[j][k][l][m]) + "\n")
        file.close()

    print("Graph " + name + "_4d_graph.txt saved!")


# function to load graph weights from file
def load_graph_weights_from_file(name, graph):
    weights_array = []
    count = 0
    if not type(name) is str:
        raise TypeError("Only strings are allowed")
    elif not type(graph) is list:
        raise TypeError("Only arrays are allowed")
    # checking if file exists then open
    if os.path.isfile("graphs/" + name + ".txt"):
        # opening file to read
        with open("graphs/" + name + ".txt", 'r') as file:
            # reading all lines of file
            lines = file.readlines()
            # for each line appending to weight array to easily set
            for line in lines:
                weights_array.append(int(line.strip()))
            # iterating through letter graph and prev_next_weights graph and
            # setting weights from weight array based on count
            for i in range(0, len(graph)-1):
                for j in range(0, len(graph[i].prev_next_weight)-1):
                    for k in range(0, len(graph[i].prev_next_weight) - 1):
                        graph[i].prev_next_weight[j][k] += weights_array[count]
                        count += 1
    else:  # creating and opening file
        file = open("graphs/" + name + ".txt", 'r')
        lines = file.readlines()
        # reading lines in file and appending to an array to easily set
        for line in lines:
            weights_array.append(int(line.strip()))
        # iterating through letter graph and prev_next_weights graph and
        # setting weights from weight array based on count
        for i in range(0, len(graph) - 1):
            for j in range(0, len(graph[i].prev_next_weight) - 1):
                for k in range(0, len(graph[i].prev_next_weight) - 1):
                    graph[i].prev_next_weight[j][k] += weights_array[count]
                    count += 1
        file.close()
    print("Loaded Graph Weights from " + name + "!")


# function to load 4d graph weights from file
def load_4d_graph_weights_from_file(name, graph):
    weights_array = []
    count = 0
    if not type(name) is str:
        raise TypeError("Only strings are allowed")
    elif not type(graph) is list:
        raise TypeError("Only arrays are allowed")
    # checking if file exists then open
    if os.path.isfile(name + ".txt"):
        # opening file to read
        with open(name + ".txt", 'r') as file:
            # reading all lines of file
            lines = file.readlines()
            # for each line appending to weight array to easily set
            for line in lines:
                weights_array.append(int(line.strip()))
            # iterating through letter graph and prev_next_weights graph and
            # setting weights from weight array based on count
            for i in range(0, len(graph) - 1):
                for j in range(0, len(graph[i].test_weights) - 1):
                    for k in range(0, len(graph[i].test_weights) - 1):
                        for l in range(0, len(graph[i].test_weights) - 1):
                            for m in range(0, len(graph[i].test_weights) - 1):
                                graph[i].test_weights[j][k][l][m] += weights_array[count]
                                count += 1
    else:  # creating and opening file
        file = open(name + ".txt", 'r')
        lines = file.readlines()
        # reading lines in file and appending to an array to easily set
        for line in lines:
            weights_array.append(int(line.strip()))
        # iterating through letter graph and prev_next_weights graph and
        # setting weights from weight array based on count
        for i in range(0, len(graph) - 1):
            for j in range(0, len(graph[i].test_weights) - 1):
                for k in range(0, len(graph[i].test_weights) - 1):
                    for l in range(0, len(graph[i].test_weights) - 1):
                        for m in range(0, len(graph[i].test_weights) - 1):
                            graph[i].test_weights[j][k][l][m] += weights_array[count]
                            count += 1
        file.close()
    print("Loaded Graph Weights from " + name + "!")


# load graphs
def load_graphs(letter_array):
    # reading in graph inputs (book text files) - uncomment to load files and load graphs
    # or add new graph from text
    # takes much longer than loading a graph from file but can be used to
    # generate a graph file if you use save file after load_graphs
    #graph = read_input_text("moby_dick", letter_array)
    #graph = read_input_text("the_blue_castle", letter_array)
    #graph = read_input_text("test", letter_array)
    graph = read_input_text("more_convos", letter_array)
    #graph = read_input_text("movie_convos", letter_array)
    #graph = read_input_text("twitter_convos", letter_array)
    #graph = read_input_text("ws_works", letter_array)
    #graph = read_input_text("program_test", letter_array)

    # Setting weights specifically for period to space letters
    for i in range(len(graph)-1):
        if i != 62:
            graph[62].add_weight(graph[77], graph[i], 10000000)
        else:
            graph[62].add_weight(graph[77], graph[i], -10000000)

    for i in range(36, 61):
        graph[77].add_weight(graph[i], graph[62], 1000000)

    # setting graph to letter array letters
    for letter in letter_array:
        letter.set_graph(graph)
    # checking if the graph is zeroed out for some reason
    if check_if_graph_zeroed(graph):
        print("GRAPH empty")
    else:
        print("Graph not empty")
    return graph


# load graphs 4d
def load_graphs_4d(letter_array):
    # reading input to 4d array of weights
    # TODO uncomment for more graphs
    graph = read_input_4d("moby_dick", letter_array)
    graph = read_input_4d("movie_convos", graph)
    graph = read_input_4d("twitter_convos", graph)
    graph = read_input_4d("the_blue_castle", graph)
    graph = read_input_4d("ws_works", graph)
    graph = read_input_4d("test", graph)


    # Setting weights specifically for period to space letters
    for i in range(len(graph) - 1):
        for j in range(len(graph)-1):
            for k in range(len(graph)-1):
                if i != 62:
                    graph[62].add_weight_4d(graph[77], graph[j], graph[i], graph[k], 10000000)
                else:
                    graph[62].add_weight_4d(graph[77], graph[j], graph[i], graph[k], -10000000)

    for i in range(36, 61):
        for j in range(36, 61):
            for k in range(36, 61):
                graph[77].add_weight_4d(graph[i], graph[j], graph[62], graph[k], 1000000)

    # setting graph to letter array letters
    for letter in letter_array:
        letter.set_graph(graph)
    # checking if the graph is zeroed out for some reason
    if check_if_graph_zeroed(graph):
        print("GRAPH empty")
    else:
        print("Graph not empty")
    return graph


# function to chat
def chat(graph, word, letter):
    word = word
    run = True
    while run:
        next_letter = letter.get_next_letter_no_random(letter.get_prev_letter(), False)
        word += letter.get_name()
        # print(word)
        while next_letter.get_name() == letter.get_name() and letter.get_name() == letter.get_prev_letter().get_name():
            next_letter = generate_random_letter(graph)
            next_letter.set_previous(letter)
        if next_letter.get_name() == '.' or next_letter.get_name() == '!' or next_letter.get_name() == '?' \
                or next_letter.get_name() == ' ':
            return word
        return generate_non_random_word_by_letter(next_letter, word, graph)
    return word


# function to add weights to letter graph
def read_input_graph_weights(filename, graph, letter_array):
    text = ""
    if not type(filename) is str:
        raise TypeError("Only strings are allowed")
    elif not type(graph) is Graph:
        raise TypeError("Only graphs are allowed")

    with open("text-files/" + filename + ".txt", encoding="utf8") as file:
        text = file.readlines()

    sentence_string = ""
    array_of_sentences = []

    prev_word = ""
    text = ' '.join(text)
    for index, word in enumerate(text):
        if index == 0 and text[index] is not None:
            let_num_next = get_number_from_letter(text[index + 1])
            let_num_prev = get_number_from_letter(77)
            graph.add_edge(text[index], text[index+1], 1, True)
            if graph.get_vertex(text[index + 1]) is None:
                graph.get_vertex([index]).add_neighbor(text[index+1], 1)
            else:
                graph.get_vertex(text[index]).add_more_weight(1, graph.get_vertex(text[index+1]))
        elif index == len(text):
            let = text[index - 1]
            let_num = get_number_from_letter(let)
        elif index != len(text) - 1:
            let_num_next = get_number_from_letter(text[index + 1])
            let_num_prev = get_number_from_letter(text[index - 1])
            graph.add_edge(text[index], text[index+1], 1, True)
            if graph.get_vertex(text[index + 1]) is None:
                graph.get_vertex([index]).add_neighbor(text[index+1], 1)
            else:
                graph.get_vertex(text[index]).add_more_weight(1, graph.get_vertex(text[index+1]))

            sentence_string += word
            if word == '.' or word == '?' or word == '!' and text[index + 1] != '.':
                array_of_sentences.append(Sentence(sentence_string, None, letter_array))
                sentence_string = ""
        sentence_string += " "
    return graph


# function to test weights and letter generation
def test_weights_and_letters(graph):
    #for i in range(0, 97):
    #    print("Graph at " + str(i) + "\n", graph[i].print_weights())
    #print(graph[15].print_weights())
    if check_if_graph_zeroed(graph):
        print("GRAPH empty")
    else:
        print("Graph not empty")
    first_letter = generate_first_letter(graph)
    first_letter.set_previous(graph[77])
    #first_letter = graph[18]
    #first_letter.add_weight(graph[30], graph[28], 1)
    #graph[18].print_weights(10)
    if first_letter.check_weights_zero():
        print("Zeroed")
    next_letter = first_letter.get_next_letter(graph[77], False)
    if first_letter.check_weights_zero():
        print("Zeroed")
    #graph[18].print_weights(10)
    #for i in graph:
    #    if not i.check_weights_zero():
    #        first_letter = i
    print("Printing letter : " + first_letter.get_name())
    print("Printing next letter : " + next_letter.get_name())
    generate_sentence_by_letter(first_letter, "", graph)


# path to target function
def path_to_target(start, goal, g):
    dijkstra(g, g.get_vertex(start), g.get_vertex(goal))
    target = g.get_vertex(goal)
    path = [target.get_id()]
    shortest(target, path)
    return path[::-1], target.get_distance()


# function to test graph functionality
def test_graph(letter_array):
    graph = Graph()
    generate_letter_graph(graph, letter_array)
    read_input_graph_weights("movie_convos", graph, letter_array)
    start, goal = 'a', '.'
    target_path, cost = path_to_target(start, goal, graph)
    print('The shortest path : {0} and costs {1}.'.format(target_path, cost))


# test function
def test_main():
    '''
    designed to test functionality of string and array parsing
    '''
    letter_array = initialize_letter_array()
    for letter in letter_array:
        print(letter.get_name())

    sentence = ["that", "is", "a", "dog", "."]
    sentence_test = "That is a dog."

    sentence1 = Sentence(sentence, None, letter_array)
    for word in sentence1.get_words():
        print(word.get_name())
    sentence2 = Sentence(sentence_test, None, letter_array)
    for word in sentence2.get_words():
        print(word.get_name())
    sentence3 = Sentence(None, sentence, letter_array)
    for word in sentence3.get_words():
        print(word.get_name())
    print(sentence3.get_words())
    print(sentence2.get_words())
    print(sentence1.get_words())

    pretend_user_input = "I'm a cowboy!"

    user_sentence = Sentence(pretend_user_input, None, letter_array)
    for word in user_sentence.get_words():
        print(word.get_name())

    pretend_user_input = ["It's", "that", "Texas'", "weather!"]
    testing_input = ["@", "~", "#", "5", "%", "^", "&", "*",
                     "(", ")", "-", "+", "="]
    testing_string_input = "@#$%^&*()_-+=![dfd]{}\\//<>dfd., " \
                           "dfdddf 3l2kjl dlfjdlo393df dddfhhh...."
    user_sentence = Sentence(None, pretend_user_input, letter_array)
    for word in user_sentence.get_words():
        print(word.get_name())

    test_sentence = Sentence(None, testing_input, letter_array)
    test_string_sentence = Sentence(testing_string_input, None, letter_array)
    #for word in test_sentence.get_words():
    #    print(word.get_name())

    pretend_user_input = "I'm a cowboy!"
    new_test_string = Sentence(pretend_user_input, None, letter_array)
    testing_sentence = ""
    for word in test_string_sentence.get_words():
        print("This is the current word: " + str(word.get_name()))
    for word in test_string_sentence.get_words():
        testing_sentence += word.get_name()
    print(testing_sentence)
    print(test_string_sentence.get_name())
    print(new_test_string.get_name())
    print(" I'm a cowboy COUNT " + str(new_test_string.get_count()))
    for word in new_test_string.get_words():
        print(word.get_name())

    test_string = "I'm new here-; my- name is-T'Challa! ' Or challa' '"
    test_sentence = Sentence(test_string, None, letter_array)
    for word in test_sentence.get_words():
        print(word.get_name())
# end test_main()


if __name__ == '__main__':
    test_main()
