#!/usr/bin/env python

from rubikview import *

def main():
    app = QApplication(sys.argv)    # Create a Qt application

    p = Partition.create([8, 8, 8])
    p.div([2,2,2])
    p.traverse_cells(assign_flat_index_gradient_color)
#    p.tilt(1,0,1)
    p.zigzag(1,0,1,1)
    p.zigzag(2,1,1,1)
    p.zigzag(0,2,1,1)

#    q = Partition.create([8,8,8])
#    q.zigzag(1,0,2,2)
#    q.tilt(1,0,1)
#    q.tile([8,2,8])
#    q.map(p)

#    for child in q:
#        child.tilt(0,1,1)
#        child.tilt(0,2,1)

    mainwindow = QMainWindow()

#    renderer = make_nested_faces
#    renderer = make_leaf_faces
    renderer = make_colored_faces
    glview = RubikView(p, renderer, mainwindow)

    mainwindow.setCentralWidget(glview)
    mainwindow.resize(800, 600)
    mainwindow.move(30, 30)

    mainwindow.show()
    app.exec_()    # Enter Qt application main loop


if __name__ == "__main__":
    main()
    sys.exit()
