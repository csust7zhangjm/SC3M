import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import rc
from .draw_utils import COLOR, MARKER_STYLE


# COLOR = ((1, 0, 0),
#          (0, 1, 0),
#          (1, 0, 1),
#          (1, 1, 0),
#          (0  , 162/255, 232/255),
#          (0.5, 0.5, 0.5),
#          (0, 0, 1),
#          (0, 1, 1),
#          (136/255, 0  , 21/255),
#          (255/255, 127/255, 39/255),
#          (0, 0, 0))

# LINE_STYLE = ['-', '--', ':', '-', '--', ':', '-', '--', ':', '-']

# MARKER_STYLE = ['o', 'v', '<', '*', 'D', 'x', '.', 'x', '<', '.']


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def draw_eao(result):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    angles = np.linspace(0, 2*np.pi, 8, endpoint=True)

    attr2value = []
    for i, (tracker_name, ret) in enumerate(result.items()):
        value = list(ret.values())
        attr2value.append(value)
        value.append(value[0])
    attr2value = np.array(attr2value)
    max_value = np.max(attr2value, axis=0)
    min_value = np.min(attr2value, axis=0)
    for i, (tracker_name, ret) in enumerate(result.items()):
        value = list(ret.values())
        value.append(value[0])
        value = np.array(value)
        value = ((value - min_value) / (max_value - min_value)) * 0.3 + 0.7
        # value = np.power(value,2.5)
        # print(len(angles))
        # print(len(value))
        # print(MARKER_STYLE[i])
        # print(i)
        plt.plot(angles, value, linestyle='-', color=COLOR[i], marker=MARKER_STYLE[i],
                label=tracker_name, linewidth=1.5, markersize=6)

    # attrs = ["All", "Background Clutters","Deformation","Fast Motion"
    #          "Illumination Variation","Plane Rotation","Low Resolution"
    #          "Motion Blur","Occlusion","Out of Plane Rotation","Out of View"
    #          "Scale Variation"] #OTB100
    attrs = ["Overall", "Camera motion",
             "Illumination change", "Motion Change",
             "Size change", "Occlusion",
             "Unassigned"] #VOT2016
    attr_value = []
    for attr, maxv, minv in zip(attrs, max_value, min_value):
        attr_value.append(attr + "\n({:.3f},{:.3f})".format(minv, maxv))
    ax.set_thetagrids(angles[:-1] * 180/np.pi, attr_value)
    ax.spines['polar'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.04), frameon=False,ncol = 5)
    ax.grid(b=False)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    plt.savefig("VOT2018_eao.pdf")
    plt.show()

if __name__ == '__main__':
    result = pickle.load(open("../../result.pkl", 'rb'))
    draw_eao(result)
