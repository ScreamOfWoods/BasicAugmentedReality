#!/usr/bin/env python

import pygame
# from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import cv2
from model_init import *
import ShaderLoader as loader
from PIL import Image
import numpy
import pyrr

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
qrDecoder = cv2.QRCodeDetector()
coords = []


class Wall(object):
    distance = 0
    x_axis = 0
    y_axis = 0
    vertices = (
        (80,-80,-1),
        (80,-80,1),
        (-80,-80,1),
        (-80,-80,-1),
        (80,80,-1),
        (80,80,1),
        (-80,80,1),
        (-80,80,-1)
        )
    faces = (
        (1,2,3,4),
        (5,8,7,6),
        (1,5,6,2),
        (2,6,7,3),
        (3,7,8,4),
        (5,1,4,8)
        )
    texcoord = ((1,0),(1,1),(0,1),(0,0))
    # -------------------------------------

    def __init__(self):
        self.coordinates = [0,0,0]
        # self.rubik_id = self.load_texture()

    def load_texture(self,frame):
        
        frame = frame[0:720,280:1000]
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(frame.tostring(), frame.shape[:2],"RGB")

        texture_data = pygame.image.tostring(surf,"RGB",1)
        width = 720
        height = 720

        ID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D,ID)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,width,height,0,GL_RGB,GL_UNSIGNED_BYTE,texture_data)
        return ID

    def render_scene(self,frame):
        glTranslatef(0,0,-150)   
        self.rubik_id = self.load_texture(frame)
        glBindTexture(GL_TEXTURE_2D,self.rubik_id)
        glBegin(GL_QUADS)

        for face in self.faces:
            for i,v in enumerate(face):
                glTexCoord2fv(self.texcoord[i])
                glVertex3fv(self.vertices[v -1])
        
        glEnd()
    
    def delete_texture(self):
        glDeleteTextures(self.rubik_id)


class Deer(object):
    distance = 0
    left_key = False
    right_key = False
    up_key = False
    down_key = False
    a_key = False
    s_key = False
    d_key = False
    r_key = False
    f_key = False
    x_axis = 0
    y_axis = 0

    def __init__(self):
        self.coordinates = [0, 0, 0]
        self.deer_id = ObjLoader()
        self.deer_id.load_model("res/deer/deer.obj")
        self.texture = self.load_texture()
        print(self.deer_id)

    def load_texture(self):
        text_offset = len(self.deer_id.vertex_index) * 12
        norm_offset = (text_offset + len(self.deer_id.texture_index) * 8)
        # create new shader program to render the 3D self.object
        shader = loader.compile_shader("res/shaders/vert.vs", "res/shaders/frag.fs")

        # Create new Vertex Buffer self.object to hold the vertices of the loaded model
        vbo = glGenBuffers(1)
        # Bind the current VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        # Load the vertices to the vertex self.object; Static draw - we won't make changes to the self.object
        glBufferData(GL_ARRAY_BUFFER, self.deer_id.model.itemsize * len(self.deer_id.model), self.deer_id.model, GL_STATIC_DRAW)

        # set the position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FLOAT, self.deer_id.model.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # set texture
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FLOAT, self.deer_id.model.itemsize * 2, ctypes.c_void_p(text_offset))
        glEnableVertexAttribArray(1)
        # normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.deer_id.model.itemsize * 3, ctypes.c_void_p(norm_offset))
        glEnableVertexAttribArray(2)

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        # Set texture wrapping params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # Set texture filtering params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # image = Image.open("res/deer/texture.png")
        image = Image.open("res/deer/mat.png")
        flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = numpy.array(list(flipped.getdata()), numpy.uint8)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

        # glEnable(GL_TEXTURE_2D)

        glUseProgram(shader)

        view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -3.0]))
        proj = pyrr.matrix44.create_perspective_projection_matrix(60.0, 640 / 480, 0.1, 200.0)
        model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

        view_loc = glGetUniformLocation(shader, "view")
        proj_loc = glGetUniformLocation(shader, "projection")
        model_loc = glGetUniformLocation(shader, "model")

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

        y_rot_90 = [0, 0, 1, 0,
                    0, 1, 0, 0,
                    -1, 0, 0, 0,
                    0, 0, 0, 1]

        transform_loc = glGetUniformLocation(shader, "transform")
        glUniformMatrix4fv(transform_loc, 1, GL_FALSE, y_rot_90)
        glUseProgram(0)
        return texture

    def render_scene(self, coords):
        glBindTexture(GL_TEXTURE_2D, self.texture)

        # z = (630 - math.sqrt(
        #    pow(abs(coords[3][0] - coords[0][0]), 2) + pow(abs(coords[3][1] - coords[0][1]), 2))) * -0.020
        z = (650 - math.sqrt(
            pow(abs(coords[3][0] - coords[0][0]), 2) + pow(abs(coords[3][1] - coords[0][1]), 2))) * -0.020
        x = (coords[0][0] + coords[1][0] + coords[3][0] + coords[5][0] - 360 * 4) / 130 - 8.5
        y = (coords[0][1] + coords[1][1] + coords[3][1] + coords[5][1] - 360 * 4) / -130

        rot = 90  # math.atan2(coords[3][1] - coords[0][1], coords[3][0] - coords[0][0]) * 57 + 45
        sideMul = 0.055
        glTranslatef(x * -z * sideMul, y * -z * sideMul, z)

        glRotate(self.coordinates[0], 1, 0, 0)
        glRotate(self.coordinates[1], 0, 1, 0)
        glRotate(-rot, 0, 1, 0)

        glDrawArrays(GL_TRIANGLES, 0, len(self.deer_id.vertex_index))

        glRotate(rot, 0, 1, 0)
        glRotate(-self.coordinates[1], 0, 1, 0)
        glRotate(-self.coordinates[0], 1, 0, 0)
        glTranslatef(-x * -z * sideMul, -y * -z * sideMul, -z)

    def delete_texture(self):
        glDeleteTextures(self.deer_id)


class Cube(object):
    distance = 0
    left_key = False
    right_key = False
    up_key = False
    down_key = False
    a_key = False
    s_key = False
    d_key = False
    r_key = False
    f_key = False
    x_axis = 0
    y_axis = 0
    vertices = (
        (1, -1, -1),
        (1, -1, 1),
        (-1, -1, 1),
        (-1, -1, -1),
        (1, 1, -1),
        (1, 1, 1),
        (-1, 1, 1),
        (-1, 1, -1)
    )
    faces = (
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (5, 1, 4, 8)
    )
    texcoord = ((0, 0), (1, 0), (1, 1), (0, 1))

    # -------------------------------------
    def __init__(self):
        self.coordinates = [0, 0, 0]
        self.rubik_id = self.load_texture("res/cube/rubik.png")

    def load_texture(self, filename):
        textureSurface = pygame.image.load(filename)
        textureData = pygame.image.tostring(textureSurface, "RGBA", 1)
        width = textureSurface.get_width()
        height = textureSurface.get_height()
        ID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, ID)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)
        return ID

    def render_scene(self, coords):

        glBindTexture(GL_TEXTURE_2D, self.rubik_id)
        z = (630 - math.sqrt(
            pow(abs(coords[3][0] - coords[0][0]), 2) + pow(abs(coords[3][1] - coords[0][1]), 2))) * -0.020
        x = (coords[0][0] + coords[1][0] + coords[3][0] + coords[5][0] - 360 * 4) / 130 - 8.5
        y = (coords[0][1] + coords[1][1] + coords[3][1] + coords[5][1] - 360 * 4) / -130

        rot = math.atan2(coords[3][1] - coords[0][1], coords[3][0] - coords[0][0]) * 57 + 45

        sideMul = 0.055
        glTranslatef(x * -z * sideMul, y * -z * sideMul, z)

        glRotate(self.coordinates[0], 1, 0, 0)
        glRotate(self.coordinates[1], 0, 1, 0)
        glRotate(-rot, 0, 0, 1)

        glBegin(GL_QUADS)

        for face in self.faces:
            for i, v in enumerate(face):
                glTexCoord2fv(self.texcoord[i])
                glVertex3fv(self.vertices[v - 1])

        glEnd()

        glRotate(rot, 0, 0, 1)
        glRotate(-self.coordinates[1], 0, 1, 0)
        glRotate(-self.coordinates[0], 1, 0, 0)
        glTranslatef(-x * -z * sideMul, -y * -z * sideMul, -z)

    def rotate_x(self):
        if self.coordinates[0] > 360:
            self.coordinates[0] = 0
        else:
            self.coordinates[0] += 2

    def rotate_y(self):
        if self.coordinates[1] > 360:
            self.coordinates[1] = 0
        else:
            self.coordinates[1] += 2

    def rotate_z(self):
        if self.coordinates[2] > 360:
            self.coordinates[2] = 0
        else:
            self.coordinates[2] += 2

    def move_away(self):
        self.distance -= 0.1

    def move_close(self):
        if self.distance < 3:
            self.distance += 0.1

    def move_left(self):
        self.x_axis -= 0.1

    def move_right(self):
        self.x_axis += 0.1

    def move_up(self):
        self.y_axis += 0.1

    def move_down(self):
        self.y_axis -= 0.1

    def keydown(self):
        if self.a_key:
            self.rotate_x()
        elif self.s_key:
            self.rotate_y()
        elif self.d_key:
            self.rotate_y()
        elif self.r_key:
            self.move_away()
        elif self.f_key:
            self.move_close()
        elif self.left_key:
            self.move_left()
        elif self.right_key:
            self.move_right()
        elif self.up_key:
            self.move_up()
        elif self.down_key:
            self.move_down()

    def keyup(self):
        self.left_key = False
        self.right_key = False
        self.up_key = False
        self.down_key = False
        self.a_key = False
        self.s_key = False
        self.d_key = False
        self.r_key = False
        self.f_key = False

    def delete_texture(self):
        glDeleteTextures(self.rubik_id)


def start_render():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_TEXTURE_2D)


def end_render():
    glDisable(GL_TEXTURE_2D)


def display(im, bbox):
    n = len(bbox)
    coords = []
    for i in range(n):
        cv2.line(im, tuple(bbox[i][0]), tuple(bbox[ (i+1)%n ][0]), (255,0,0), 3)
        coords.append(tuple(bbox[i][0]))
        coords.append(tuple(bbox[(i + 1) % n][0]))
    return im, coords


def main():
    pygame.init()
    pygame.display.set_mode((640,480),pygame.DOUBLEBUF|pygame.OPENGL)
    clock = pygame.time.Clock()
    done = False

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(60, 640.0/480.0, 0.1, 200.0)

    glEnable(GL_DEPTH_TEST)

    deer = Deer()
    cube = Cube()
    feed = Wall()

    # ----------- Main Program Loop -------------------------------------
    while not done:
        # --- Main event loop
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done = True # Flag that we are done so we exit this loop
        start_render()
        ret, frame = cap.read()

        data, bbox, rectified = qrDecoder.detectAndDecode(frame)
        # Render Deer or Cube depending on the information read from the QR code
        if len(data) > 0:
            frame, coords = display(frame, bbox)
            if data == "deer":
                deer.render_scene(coords)
            if data == "cube":
                cube.render_scene(coords)
        feed.render_scene(frame)
        end_render()

        pygame.display.flip()
        clock.tick(30)

    deer.delete_texture()
    pygame.quit()


if __name__ == '__main__':
    main()

