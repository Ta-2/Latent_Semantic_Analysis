import matplotlib.pyplot as plt

def illustrate(datas, margin=30):
    xmax, xmin = max([d[0] for d in datas]), min([d[0] for d in datas])
    ymax, ymin = max([d[1] for d in datas]), min([d[1] for d in datas])
    plt.scatter(x=[d[0] for d in datas], y=[d[1] for d in datas], c="blue")
    plt.xlim([xmin-margin, xmax+margin])
    plt.ylim([ymin-margin, ymax+margin])
    plt.grid()
    plt.show()