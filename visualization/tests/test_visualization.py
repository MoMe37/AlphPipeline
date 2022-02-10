import numpy.testing as npt
import numpy as np
import visualization as vis

def test_Visualization():
    title = ('', '', '')
    visu = vis.Visualization(1, 3, title)

    file_list = np.array([['', '', '']])
    npt.assert_equal(visu.filepath, file_list)

    visu.update_fig(1, 1, '../fadg0/face_mesh/sa1')
    visu.update_fig(1, 2, './prediction')
    visu.update_fig(1, 3, './prediction')
    
    file_list2 = np.array([['../fadg0/face_mesh/sa1', './prediction', './prediction']])
    npt.assert_equal(visu.filepath, file_list2)

    visu.animate()
    visu.set_camera()
    visu.afficher()

def test_vizualisation_retargeting():
    title = ('', '', '', '')
    visu = vis.Visualization(2, 2, title)

    visu.update_fig(1, 1, '../fadg0/face_mesh/sa1')
    visu.update_fig(1, 2, './prediction')
    visu.update_fig(2, 1, './prediction')

    visu.update_fig_retargeting(2, 2, './alphsistant/data/suzanne_test/prediction_retargeting.yml')

    type_list = [['normal', 'normal'], ['normal', 'retargeting']]
    npt.assert_equal(visu.type, type_list)

    visu.animate()
    visu.set_camera()
    visu.afficher()