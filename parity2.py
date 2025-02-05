from manimlib import *

config.pixel_width = 1920  # Increase width
config.pixel_height = 1080  # Increase height
config.frame_width = 16*2  # Adjust the visible width
config.frame_height = 9*2  # Adjust the visible height
print(manimlib.__file__)
import numpy as np
from scipy.sparse import csr_array
from math import log2
def is_norm_slow(state,eps = 1e-6):
    phi_0,phi_1 = state.flatten()
    abs1 = phi_0.real**2 + phi_0.imag**2
    abs2 = phi_1.real**2 + phi_1.imag**2
    return abs(1-(abs1 + abs2)) <= eps
def prob_0(state):
    return state[0].real**2 + state[0].imag**2
def prob_1(state):
    return state[1].real**2 + state[1].imag**2
q0 = np.array([[1],[0]])
q1 = np.array([[0],[1]])
H = np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
q_plus = H@q0
q_minus = H@q1
Rx = lambda phi : np.array([[np.cos(phi/2),-1j*np.sin(phi/2)],[-1j*np.sin(phi/2),np.cos(phi/2)]]) 
Ry = lambda phi : np.array([[np.cos(np.pi*phi/2),-np.sin(np.pi*phi/2)],[np.sin(np.pi*phi/2),np.cos(np.pi*phi/2)]]) 
Rz = lambda phi : np.array([[np.exp(-1j*phi/2),0],[0,np.exp(1j*phi/2)]]) 
I = np.eye(2)
X = np.array([[0.0,1.0],[1.,0.]])
Z = np.array([[1.,0.],[0.,-1.]])


def qubit_to_bloch(psi):

    alpha, beta = psi
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    return np.array([x, y, z])

image_gate_dict = {
    "X" : ("X.png", X),
    "Z" : ("Z.png", Z),
    "q0" : ("q0.png", q0),
    "q1" : ("q1.png", q1),
    "H" : ("H.png", H),
    "I" : ("I.png",I),
    "meausre" : ("measure.png",None)
}
gates_toshow = ["q0","X","H","H","X"]
def check_even(num):
    
    return Tex(f"\\text{{Apply}}" f"R_y(\\frac{{{num[0]}}}{{{num[1]}}})", font_size=36,color=RED).scale(.9)

#from manim import *

class Complex_Parity(Scene):
    def construct(self):
        
        ### for later TODO
        
        text = Tex(
            "\\text{Parity Game 2}: \\text{Sum of integers } k_1, k_2, \\dots, k_{D-1} \\in \\mathbb{Q}",
            tex_to_color_map={
                "\\text{Parity Game 2}": RED,  # Color "Parity Game" red
            }
        ).shift(UP*2)
        #formula1 = Tex(r"E = mc^2")
        self.play(Write(text))
        self.wait(1)
        #self.play(FadeOut(text))
        text = Tex("\\text{With the promise that: } \sum_{i=0}^{D-1} k_i \in\mathbb{Z}", font_size=40).next_to(text, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(text))
        
        
        
        text1 = Tex("\\text{Start in state: } |0\\rangle", font_size=40).next_to(text, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(text1))
        self.wait(1)
        text2 = Tex("\\text{Apply rotation of angle }k_i \\text{ around y-axis}", font_size=40).next_to(text1, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(text2))
        self.wait(1)
        
        self.play(FadeOut(text1),FadeOut(text2))
        parity_im = ImageMobject("extracted_images/" + "parity_2.png").scale(1).next_to(text, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(parity_im))
        self.wait(1)

        self.clear()
        text = Text("Notice how consecutive rotations can be formulated to one.").scale(.6).shift(UP*3)
        self.play(Write(text))
        parity_im = ImageMobject("extracted_images/" + "par_sum.png").scale(.5).shift(UP)
        self.play(FadeIn(parity_im))
        self.wait(2)
        self.clear()
        parity_im = ImageMobject("extracted_images/" + "par_math.png").scale(1)#.shift(UP)
        self.play(FadeIn(parity_im))
        self.wait(2)
        self.clear()

        
        numb_ar = [[2, 3], [5, 3], [8, 3],[3,3]]

        # Dynamically create the Tex string with \qquad spacing
        fractions_str = " \qquad ".join(
            [f"\\frac{{{num}}}{{{den}}}" for num, den in numb_ar]  # Generate fractions
        )
        numbers = Tex(fractions_str, font_size=48).to_edge(UP).shift(UP*.5).fix_in_frame()

        # Get the positions of each fraction
        # Split the numbers Tex object into individual substrings
        number_positions = []
        for i, (num, den) in enumerate(numb_ar):
            # Create a substring for each fraction
            fraction_substring = numbers.get_part_by_tex(f"\\frac{{{num}}}{{{den}}}")
            number_positions.append(fraction_substring.get_center())
        number_positions = number_positions
        pointer = Arrow(
            start=number_positions[-1] + DOWN*1.1,  
            end=number_positions[-1],
            color=YELLOW,
        ).fix_in_frame().shift(DOWN*.2)

        
        axes = ThreeDAxes(
            x_range=np.array([-1.3, 1.3, 1]),
            y_range=np.array([-1.2, 1.2, 1]),
            z_range=np.array([-1.2, 1.1, 1]),
            width=2, height=2, depth=1.5,
            axis_config={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "include_ticks": False
            },
        ).scale(1.5)
        z_label_top = Tex('|0\\rangle').next_to(axes.z_axis.get_end(), UP*4).fix_in_frame()
        z_label_bottom = Tex('|1\\rangle').next_to(axes.z_axis.get_start(), DOWN*5).fix_in_frame()
        #y_label_top = Tex('|i_+\\rangle').next_to(axes.y_axis.get_end(), OUT*2).fix_in_frame().shift(LEFT*1.5).shift(DOWN*1.5)
        #y_label_bottom = Tex('|i_-\\rangle').next_to(axes.y_axis.get_start(), IN*2).fix_in_frame().shift(RIGHT*2.3).shift(UP)    
        arrow_top = Arrow(
            start=z_label_top.get_right() + RIGHT * 0.5,
            end=z_label_top.get_right(),
            buff=0.1,
            color=YELLOW
        ).fix_in_frame()

        arrow_bottom = Arrow(
            start=z_label_bottom.get_right() + RIGHT * 0.5,
            end=z_label_bottom.get_right(),
            buff=0.1,
            color=YELLOW
        ).fix_in_frame()

        # Add text next to the arrows
        text_top = Text('even',font_size=30).next_to(arrow_top, RIGHT).fix_in_frame()
        text_bottom = Text('odd',font_size=30).next_to(arrow_bottom, RIGHT).fix_in_frame()
        #state_vector = qubit_to_bloch(q0)
        #vector = Vector(state_vector.squeeze(),color=RED)

        

        self.play(Write(numbers))
        # TODO
        self.wait(1)

        self.play(
            self.camera.frame.animate.set_euler_angles(
                theta=135 * DEGREES,
                phi=60 * DEGREES,
                gamma=0 * DEGREES
            ),
            run_time=3  
        )
        self.add(z_label_top,z_label_bottom)
        self.play(FadeIn(axes),Write(text_top), Write(text_bottom),GrowArrow(arrow_top), GrowArrow(arrow_bottom))
        sphere = Sphere(
            radius=1,
            resolution=(101, 51),
            color=BLUE,
            opacity=0.6,
            #gloss=0.3,
            #shadow=0.4
        )
        
        self.play(FadeIn(sphere))
        state = q0
        state_vector = qubit_to_bloch(q0)
        vector = Vector(state_vector.squeeze(),color=RED)
        self.play(FadeIn(vector)) #TODO text?
        self.wait(.5)
        label = check_even(numb_ar[-1]).fix_in_frame()
        label.next_to(pointer,DOWN).shift(UP*.4)
        self.play(GrowArrow(pointer), Write(label))
        self.wait(.5)
        print(numb_ar[-1])
        gate_ac = Ry(numb_ar[-1][0]/numb_ar[-1][1])
        #if numb_ar[-1] % 2 != 0:
        state = gate_ac @ state
        new_vector = Vector(qubit_to_bloch(state).squeeze(),color=RED)
            
        self.play(Transform(vector,new_vector))

        for i in range(len(number_positions) - 2, -1, -1): 
            new_pointer = Arrow(
            start=number_positions[i] + DOWN*1.1,  
            end=number_positions[i],
            color=YELLOW,
        ).fix_in_frame().shift(DOWN*.2)
            new_label = check_even(numb_ar[i]).next_to(new_pointer, DOWN).shift(UP*.4).fix_in_frame()

            self.play(
                Transform(pointer, new_pointer),
                Transform(label, new_label)
            )
            gate_ac = Ry(numb_ar[i][0]/numb_ar[i][1])
            #if numb_ar[-1] % 2 != 0:
            state = gate_ac @ state
            new_vector = Vector(qubit_to_bloch(state).squeeze(),color=RED)
            self.play(Transform(vector,new_vector))
            self.wait(1)
        self.wait(1)
        
        
        self.play(
                sphere.animate.set_opacity(0.2),
                axes.animate.set_opacity(0.2),
                z_label_top.animate.set_opacity(0.2),
                z_label_bottom.animate.set_opacity(0.2),
                vector.animate.set_opacity(0.2),
                text_top.animate.set_opacity(0.2),
                text_bottom.animate.set_opacity(0.2),
                FadeOut(pointer),FadeOut(label),FadeOut(arrow_top),FadeOut(arrow_bottom)
                )
        is_ev = "even" if np.abs(np.abs(state[0])-1)<=1e-4 else "odd"
        print(state[0])
        text_ac = Text(f"Deterministic Readout -> {is_ev}" ,font_size=40).fix_in_frame()
        self.play(Write(text_ac))
        numbers_sum = Text(f"Actual sum: {np.sum(np.array(numb_ar),axis=0)[0] / numb_ar[0][1]}", font_size=48).fix_in_frame()
        numbers_sum.to_edge(UP)  # Position at the top of the screen
        self.play(Transform(numbers,numbers_sum))
        self.play(FadeOut(text_ac))


        
        self.clear()
