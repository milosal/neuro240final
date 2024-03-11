from hfunction import H 
import matplotlib.pyplot as plt

def graph_all_up_to_n():
    limit = int(input("Up to what number would you like to see the sequence lengths? "))
    x = []
    i = 1
    while i <= limit:
        x.append(i)
        i += 1

    y = []
    for value in x:
        n_h = H(value)
        y.append(n_h)
    
    plt.ion()
    plt.xlabel("n")
    plt.ylabel("n_h")
    plt.scatter(x, y, marker=".")
    plt.grid(True)
    plt.show()

    input("Press enter to close the plot... ")

    plt.close()

graph_all_up_to_n()