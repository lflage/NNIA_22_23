def my_print():
    print("Hello Tim!!")

def my_mean(a):
    # using python built in functions to calculate the mean
    # I could also iterate over the np.array and sum the items
    # but its ineficcient
    return sum(a)/len(a)

def my_std(a):
    mean = my_mean(a)
    b = 0
    for i in a:
        b += (i - mean)**2
    return (b / len(a - 1))**(1/2)
