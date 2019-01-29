from OpenGL.GL import *
import OpenGL.GL.shaders

#Helper file to load and compile shaders

#Load the shader from raw source
def load_shader(shader_file):
    shader_source = ""
    with open(shader_file) as f:
        shader_source = f.read()
    f.close()
    return str.encode(shader_source)

#Compile shader program from Vertex and Fragment (Pixel) shader
def compile_shader(vs, fs):
    vert_shader = load_shader(vs) #Send the vertex shader source to be read and encoded
    frag_shader = load_shader(fs) #Send the fragment shader source to be read and encoded

    #Compile the shader program from FS and VS to be used for rendering
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vert_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(frag_shader, GL_FRAGMENT_SHADER))
    return shader
