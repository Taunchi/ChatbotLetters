# dboyer2@live.nmhu.edu
# Program to chat with using letter trees
# convos obtained here https://github.com/Phylliida/Dialogue-Datasets
from Letters import *
import time


# prompt function to get input
def prompt():
    usr_input = input("\n Would you like to generate a paragraph? (y/n)\n ")
    if usr_input == 'y':
        return True
    else:
        return False


# prompt function to get input
def prompt2():
    usr_input = input("\n Would you like to generate a random sentence? (y/n) \n")
    if usr_input == 'y':
        return True
    else:
        return False


# prompt function to get input
def prompt3():
    usr_input = input("\n Would you like to generate a sentence? (y/n) \n")
    if usr_input == 'y':
        return True
    else:
        return False


# TODO change case 1 to use user input
# prompt function to choose input
def prompt4(graph):
    start = 'y'
    if not type(graph) is list:
        raise TypeError("Only arrays are allowed")
    while start == 'y':
        start = input(" Would you like to continue with the program? (y/n)\n ")
        usr_input = input(" What would you like to do? \n 1. Chat"
                          "\n 2. Generate Sentence. \n 3. Generate Random Sentence. "
                          "\n 4. Generate Paragraph \n 5. Load Additional Graph \n "
                          "6. Generate Code - WILL break Program - Need more training data\n"
                          " 7. Generate 4D sentence - VERY REMARKABLY SLOW \n"
                          " 8. Quit Program \n")
        match usr_input:
            case '1':
                while prompt6(graph):
                    print("\n ")
            case '2':
                while prompt3():
                    print(generate_non_random_sentence_by_word(7, graph, '.'))
            case '3':
                while prompt2():
                    print(generate_sentence_by_word(7, graph, '.'))
            case '4':
                while prompt():
                    for sentence in generate_paragraph(graph):
                        print(sentence + " ")
            case '5':
                while prompt5():
                    graph_choice = input(" Which graph would you like to add? \n 1. Moby Dick"
                                         "\n 2. The Blue Castle \n 3. Shakespeare Works "
                                         "\n 4. Movie Convos \n 5. More Convos \n ")
                    match graph_choice:
                        case '1':
                            load_graph_weights_from_file("moby_dick_graph", graph)
                        case '2':
                            load_graph_weights_from_file("the_blue_castle_graph", graph)
                        case '3':
                            load_graph_weights_from_file("ms_works_graph", graph)
                        case '4':
                            load_graph_weights_from_file("movie_convos_graph", graph)
                        case '5':
                            load_graph_weights_from_file("more_convos_graph", graph)
                        case _:
                            print("That is not a valid option")
            case '6':
                print(generate_code_by_letter(generate_random_letter(graph), "", graph))
            case '7':
                while prompt7():
                    print(generate_4d_sentence(graph))
            case '8':
                usr_input = 'n'
                print("\n See ya \n")
            case _:
                print("\n " + usr_input + " is not a valid option. Bye\n")


# function prompt to change or add graph
def prompt5():
    usr_input = input("\n Would you like to choose a graph? (y/n) \n")
    if usr_input == 'y':
        return True
    else:
        return False


# prompt function to chat with user
def prompt6(graph):
    usr_input = input("\n Type something and I'll reply. (type exit to stop)\n\n ")
    read_user_input(usr_input, graph)
    print(generate_chat(random.randint(5, 10), graph, '.', graph[get_number_from_letter(usr_input[len(usr_input) - 1])]))
    if usr_input != 'exit':
        return True
    else:
        return False


# prompt function to generate 4d sentence
def prompt7():
    usr_input = input("\n Would you like to generate a 4D sentence? (y/n) \n")
    if usr_input == 'y':
        return True
    else:
        return False


# main function
def main():
    letter_array = initialize_letter_array()
    # load_graphs and save_graph work together to parse file weights
    # and save graph from text file
    # loading and parsing graphs from input text files
    #graph = load_graphs(letter_array)
    graph = load_graphs_4d(letter_array)

    # saving graph to file - change name to match text files you're loading
    # in the load_graphs() function - 4d graph too large to save
    #save_graph(graph, "more_convos")
    #save_4d_graph(graph, "movie_convos")

    # loading default graph weights from text file - uncomment to load
    #load_graph_weights_from_file("program_test_graph", letter_array)
    load_graph_weights_from_file("movie_convos_graph", letter_array)
    load_graph_weights_from_file("more_convos_graph", letter_array)
    #load_graph_weights_from_file("twitter_convos_graph", letter_array)
    load_graph_weights_from_file("ms_works_graph", letter_array)
    load_graph_weights_from_file("moby_dick_graph", letter_array)

    #load_4d_graph_weights_from_file("movie_convos", letter_array)

    while True:
        #print(generate_4d_sentence(graph))
        print(generate_4d_sentence_2letter(graph))
    # prompting the user with options in a loop
    #prompt4(graph)
# end main():


if __name__ == '__main__':
    main()
