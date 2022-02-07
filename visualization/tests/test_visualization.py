import numpy.testing as npt
import numpy as np
import visualization as vis

def test_Visualization():
    visu = vis.Visualization(1, 3)

    file_list = np.array([['', '', '']])
    npt.assert_equal(visu.filepath, file_list)

    visu.update_fig(1, 1, './prediction')
    visu.update_fig(1, 2, './prediction')
    visu.update_fig(1, 3, './prediction')
    
    file_list2 = np.array([['./prediction', './prediction', './prediction']])
    npt.assert_equal(visu.filepath, file_list2)

    visu.afficher()