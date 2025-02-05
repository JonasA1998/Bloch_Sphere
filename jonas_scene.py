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
I = np.eye(2)
X = np.array([[0.0,1.0],[1.,0.]])
Z = np.array([[1.,0.],[0.,-1.]])
Rx = lambda phi : np.array([[np.cos(phi/2),-1j*np.sin(phi/2)],[-1j*np.sin(phi/2),np.cos(phi/2)]]) 
Ry = lambda phi : np.array([[np.cos(phi/2),-np.sin(phi/2)],[np.sin(phi/2),np.cos(phi/2)]]) 
Rz = lambda phi : np.array([[np.exp(-1j*phi/2),0],[0,np.exp(1j*phi/2)]]) 
T = np.array([[1,0],[0,np.exp(np.pi*1j/4)]])
S = np.array([[1,0],[0,np.exp(np.pi*1j/2)]])
def conc_gate(gates):
    return np.linalg.multi_dot(gates)
Y = conc_gate([S,H,Z,H,S,S,S])
t_state = Rx(np.pi/2)@q0
print(t_state)

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
    "q1" : ("q1(1).png", q1),
    "H" : ("H.png", H),
    "I" : ("I.png",I),
    "S" : ("S(1).png",S),
    "T" : ("T(1).png",T),
    "meausre" : ("measure.png",None),
    "q_plus" : (None,q_plus)
}
gates_toshow = ["q0","X","H","H","X"]
def check_even(num):
    if num % 2 == 0:
        return "\\text{The number is even, do nothing!}"
    else:
        return "\\text{The number is odd add an X gate!}"
def write(text,red=None):
    if red is not None:
        return Tex(f"\\text{{{text}}}",tex_to_color_map={
                f"\\text{{{red}}}": RED,  # Color "Parity Game" red
            })
    else:
        return Tex(f"\\text{{{text}}}")
    

states2 = [
    (r"|0\rangle", r"\begin{bmatrix} 1 \\ 0 \end{bmatrix}"),
    (r"|1\rangle", r"\begin{bmatrix} 0 \\ 1 \end{bmatrix}"),
    (r"|+\rangle", r"\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}"),
    (r"|-\rangle", r"\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}"),
    (r"|i+\rangle", r"\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ i \end{bmatrix}"),
    (r"|i-\rangle", r"\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -i \end{bmatrix}"),
]

gate_to_str = {"q0":r"|0\rangle","q1":r"|1\rangle"}

#from manim import *
class Intro_Univ(Scene):
    def construct(self):
        
        
        text = write("Introduction: Bloch-Sphere on elementary gates").shift(UP*2)
        text2 = write("Recall important states:","important states").next_to(text, DOWN, aligned_edge=LEFT)
        self.play(Write(text))
        self.play(Write(text2))
        for i, (state2, vector2) in enumerate(states2):
            if i % 2 == 0:  
                row = 0
                col = i // 2
            else:  
                row = 1
                col = i // 2

            tex = Tex(
                rf"{state2} = {vector2}",
                font_size=40,
            ).shift(RIGHT * col * 4 + DOWN * row * 1.5).shift(LEFT * 5)
            self.play(Write(tex))
            self.wait(1)
        self.wait(1)
        self.clear()
                    
        self.wait(1)
        text = write("Let us start with the elementary gates!").shift(UP*2)
        self.play(Write(text))
        self.wait(1)
        self.clear()
        axes = ThreeDAxes(
            x_range=np.array([-1.3, 1.3, 1]),
            y_range=np.array([-1.2, 1.2, 1]),
            z_range=np.array([-1.2, 1.1, 1]),
            width=3, height=3, depth=3,
            axis_config={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "include_ticks": False
            },
        ).scale(1.5)
        #axes.add_axis_labels('|+\\rangle','|i+\\rangle','|0\\rangle',font_size=60)
        x_label_top = Tex('|-\\rangle').next_to(axes.x_axis.get_end(), LEFT*1).fix_in_frame().shift(UP*.9)
        x_label_bottom = Tex('|+\\rangle').next_to(axes.x_axis.get_start(), RIGHT*.05).fix_in_frame().shift(DOWN)
        z_label_top = Tex('|0\\rangle').next_to(axes.z_axis.get_end(), UP*7).fix_in_frame()
        z_label_bottom = Tex('|1\\rangle').next_to(axes.z_axis.get_start(), DOWN*9).fix_in_frame()
        y_label_top = Tex('|i_{-}\\rangle').next_to(axes.y_axis.get_end(), OUT*2).fix_in_frame().shift(LEFT*1.5).shift(DOWN*1.5)
        y_label_bottom = Tex('|i_+\\rangle').next_to(axes.y_axis.get_start(), IN*2).fix_in_frame().shift(RIGHT*2.3).shift(UP)    
        
        self.play(
            self.camera.frame.animate.set_euler_angles(
                theta=135 * DEGREES,
                phi=60 * DEGREES,
                gamma=0 * DEGREES
            ),
            run_time=3  
        )
        self.add(x_label_top,x_label_bottom,z_label_top,z_label_bottom,y_label_top,y_label_bottom)
        self.play(FadeIn(axes))
        
        
        sphere = Sphere(
            radius=1,
            resolution=(101, 51),
            color=BLUE,
            opacity=0.8,
            #gloss=0.3,
            #shadow=0.4
        ).set_z_index(1)
        
        self.wait(1)
        self.play(FadeIn(sphere))
        
        #self.set_camera_orientation(phi=60 * DEGREES, theta=135 * DEGREES)
        
        box = Rectangle(width=5.2, height=1.6, color=RED).to_corner(UL, buff=1.0).fix_in_frame().set_fill(color=WHITE, opacity=1).shift(UP)
        #box_text = Text("", font_size=24).next_to(box.get_top(), DOWN, buff=0.3).fix_in_frame()
        #self.add_fixed_in_frame_mobjects(box)  # Keeps the box fixed in the frame
        self.add(box)
        #self.fix_in_frame(box, box_text)
        #self.add_fixed_in_frame_mobjects(characters)

        self.wait(2)
        said = False
        gates_toshow_list = [["q0","X"],["q1","X"],["q0","H"],["q1","H"],["q0","X","H"],["q0","X","H","X"],["q0","H","X"],["q0","H","X","H"],["q0","H","X","H","X"]]
        for gates_toshow in gates_toshow_list:
            if said == False and len(gates_toshow)>2:
                self.play(
                sphere.animate.set_opacity(0.2),
                axes.animate.set_opacity(0.2),
                x_label_top.animate.set_opacity(0.2),
                x_label_bottom.animate.set_opacity(0.2),
                y_label_top.animate.set_opacity(0.2),
                y_label_bottom.animate.set_opacity(0.2),
                z_label_top.animate.set_opacity(0.2),
                z_label_bottom.animate.set_opacity(0.2),
                )
                text_ac = Tex(f"\\text{{Now the combination (Only for state:}}{gate_to_str['q0']})" ,font_size=60).fix_in_frame()
                self.play(Write(text_ac))
                self.wait(1)
                self.play(FadeOut(text_ac),
                        sphere.animate.set_opacity(0.8),
                axes.animate.set_opacity(1),
                x_label_top.animate.set_opacity(1),
                x_label_bottom.animate.set_opacity(1),
                y_label_top.animate.set_opacity(1),
                y_label_bottom.animate.set_opacity(1),
                z_label_top.animate.set_opacity(1),
                z_label_bottom.animate.set_opacity(1),
                )
                said =True
            elif len(gates_toshow) == 2:
                self.play(
                sphere.animate.set_opacity(0.2),
                axes.animate.set_opacity(0.2),
                x_label_top.animate.set_opacity(0.2),
                x_label_bottom.animate.set_opacity(0.2),
                y_label_top.animate.set_opacity(0.2),
                y_label_bottom.animate.set_opacity(0.2),
                z_label_top.animate.set_opacity(0.2),
                z_label_bottom.animate.set_opacity(0.2),
                )
                text_ac = Tex(f"\\text{{Apply }}{gates_toshow[1]}\\text{{ onto }}{gate_to_str[gates_toshow[0]]}" ,font_size=60).fix_in_frame()
                self.play(Write(text_ac))
                self.wait(1)
                self.play(FadeOut(text_ac),
                        sphere.animate.set_opacity(0.8),
                axes.animate.set_opacity(1),
                x_label_top.animate.set_opacity(1),
                x_label_bottom.animate.set_opacity(1),
                y_label_top.animate.set_opacity(1),
                y_label_bottom.animate.set_opacity(1),
                z_label_top.animate.set_opacity(1),
                z_label_bottom.animate.set_opacity(1),
                )

            ### start state
            im_path , state = image_gate_dict[gates_toshow.pop(0)]
            state_vector = qubit_to_bloch(state)
            vector = Vector(state_vector.squeeze(),color=RED) #TODO tip_shape: "https://docs.manim.community/en/stable/reference/manim.mobject.geometry.tips.ArrowTip.html"
            vector.set_color(RED)
            image_start = ImageMobject("gate_imgs/" + im_path)
            image_start.scale(0.3).fix_in_frame()
            image_start.to_edge(UP).move_to(box.get_center()).shift(RIGHT*2)
            self.add(vector,image_start)
            self.play(FadeIn(vector),FadeIn(image_start))
            vector_position_text = Tex(
                f"\\text{{Current State: }} "
                f"\\begin{{bmatrix}} {np.round(state[0].item(),2)} \\\\ {np.round(state[1].item(),2)} \\end{{bmatrix}}",
                font_size=40
            ).fix_in_frame().to_corner(UR).shift(LEFT)
            self.play(Write(vector_position_text))
            shown_images = []
            for i, gate_name in enumerate(gates_toshow):
                im_path , gate = image_gate_dict[gate_name]
                state = gate @ state
                #vector_position_text.become(Text(f"Current State: {np.round(state,2)}", font_size=24)).fix_in_frame().to_corner(UR)
                vector_position_text2 = Tex(
                f"\\text{{Current State: }} "
                f"\\begin{{bmatrix}} {np.round(state[0].item(),2)} \\\\ {np.round(state[1].item(),2)} \\end{{bmatrix}}",
                font_size=40
            ).fix_in_frame().to_corner(UR).shift(LEFT)
                vector_position_text.become(vector_position_text2)
                #self.play(Transform(vector_position_text, vector_position_text2))
                state_vector = qubit_to_bloch(state)
                vector_new = Vector(state_vector.squeeze()).set_color(RED)
                image = ImageMobject("gate_imgs/" + im_path)
                shown_images.append(image)
                image.scale(0.3).fix_in_frame()
                image.to_edge(UP).move_to(box.get_center()).shift(RIGHT*2).shift(LEFT * ((i+1) / 1))
                self.add(vector,image)
                self.play(Transform(vector, vector_new),FadeIn(image))
                self.wait(1)
            
            ## remove things for next iterations
            self.play(FadeOut(vector),FadeOut(vector_position_text),FadeOut(vector_new),FadeOut(image_start),*[FadeOut(image) for image in shown_images])

        self.wait(1)

        self.play(
                sphere.animate.set_opacity(0.2),
                axes.animate.set_opacity(0.2),
                x_label_top.animate.set_opacity(0.2),
                x_label_bottom.animate.set_opacity(0.2),
                y_label_top.animate.set_opacity(0.2),
                y_label_bottom.animate.set_opacity(0.2),
                z_label_top.animate.set_opacity(0.2),
                z_label_bottom.animate.set_opacity(0.2),
                )
        text_ac = Text("Overall these 2 gates form 8 functionalities" ,font_size=40).fix_in_frame()
        self.play(Write(text_ac))
        self.play(FadeOut(text_ac))
        text_ac = Text("Recall the Clifford gates, induced by H and S" ,font_size=40).fix_in_frame()
        self.play(Write(text_ac))
        self.wait(1)
        text_ac2 = Text("They form a total of 24 different functionalities" ,font_size=30).fix_in_frame().next_to(text_ac, DOWN, aligned_edge=LEFT)
        self.play(Write(text_ac2))
        self.wait(1)
        self.play(FadeOut(text_ac),FadeOut(text_ac2))

        image_univ = ImageMobject("extracted_images/" + "univ_gate.png").fix_in_frame().scale(0.55)
        self.play(FadeIn(image_univ))
        self.wait(8)
        self.play(FadeOut(image_univ))

        text_ac = Text("Example 1:" ,font_size=40).fix_in_frame()
        self.play(Write(text_ac))
        equation = Tex(
            r"R_X\left(\frac{\pi}{2}\right) \approx H \cdot S \cdot H",
            color=WHITE
        ).fix_in_frame().next_to(text_ac, DOWN, aligned_edge=LEFT).shift(LEFT)
        
        equation.set_color_by_tex("R_X", RED)  
        equation.set_color_by_tex("H", BLUE)   
        equation.set_color_by_tex("S", GREEN)  
        self.play(Write(equation))
        self.wait(5)
        self.play(FadeOut(text_ac),FadeOut(equation),
                        sphere.animate.set_opacity(0.8),
                axes.animate.set_opacity(1),
                x_label_top.animate.set_opacity(1),
                x_label_bottom.animate.set_opacity(1),
                y_label_top.animate.set_opacity(1),
                y_label_bottom.animate.set_opacity(1),
                z_label_top.animate.set_opacity(1),
                z_label_bottom.animate.set_opacity(1),
                )


        # readout
        state_vector = qubit_to_bloch(Rx(np.pi/2)@q0)
        vector_t = Vector(state_vector.squeeze(),color=RED,stroke_width=10)
        self.play(FadeIn(vector_t))
        label = Text("target", color=RED,font_size=35).next_to(vector_t.get_center()*0.8, UP).rotate(vector_t.get_angle()).shift(LEFT*0.25)
        self.play(FadeIn(label))
        self.wait(1)
        #------------------------------------
        shown_images = []
        gates_toshow = ["q0","H","S","H"]
        im_path , state = image_gate_dict[gates_toshow.pop(0)]
        state_vector = qubit_to_bloch(state)
        vector = Vector(state_vector.squeeze(),color=RED) #TODO tip_shape: "https://docs.manim.community/en/stable/reference/manim.mobject.geometry.tips.ArrowTip.html"
        vector.set_color(RED)
        image = ImageMobject("gate_imgs/" + im_path)
        shown_images.append(image)
        image.scale(0.3).fix_in_frame()
        image.to_edge(UP).move_to(box.get_center()).shift(RIGHT*2)
        self.add(vector,image)
        self.play(FadeIn(vector),FadeIn(image))
        vector_position_text = Tex(
            f"\\text{{Current State: }} "
            f"\\begin{{bmatrix}} {np.round(state[0].item(),2)} \\\\ {np.round(state[1].item(),2)} \\end{{bmatrix}}",
            font_size=40
        ).fix_in_frame().to_corner(UR).shift(LEFT)
        self.play(Write(vector_position_text))
        t_state = Rx(np.pi/2)@q0
        print(t_state)
        
        for i, gate_name in enumerate(gates_toshow):
            im_path , gate = image_gate_dict[gate_name]
            state = gate @ state
            #vector_position_text.become(Text(f"Current State: {np.round(state,2)}", font_size=24)).fix_in_frame().to_corner(UR)
            vector_position_text2 = Tex(
            f"\\text{{Current State: }} "
            f"\\begin{{bmatrix}} {np.round(state[0].item(),2)} \\\\ {np.round(state[1].item(),2)} \\end{{bmatrix}}",
            font_size=40
        ).fix_in_frame().to_corner(UR).shift(LEFT)
            
            
            vector_position_text.become(vector_position_text2)
            #self.play(Transform(vector_position_text, vector_position_text2))
            state_vector = qubit_to_bloch(state)
            vector_new = Vector(state_vector.squeeze()).set_color(RED)
            image = ImageMobject("gate_imgs/" + im_path)
            shown_images.append(image)
            image.scale(0.3).fix_in_frame()
            image.to_edge(UP).move_to(box.get_center()).shift(RIGHT*2).shift(LEFT * ((i+1) / 1))
            self.add(vector,image)
            self.play(Transform(vector, vector_new),FadeIn(image))
            self.wait(1)

        self.wait(1)

        self.play(FadeOut(label),FadeOut(vector_t),FadeOut(vector),FadeOut(vector_position_text),FadeOut(vector_new),*[FadeOut(image) for image in shown_images],FadeOut(box))
        self.wait(1)
        self.clear()
        # #----------------------------------
        # #RX(π/3)≈H⋅T⋅S⋅T⋅H⋅S
        self.play(
                sphere.animate.set_opacity(0.2),
                axes.animate.set_opacity(0.2),
                x_label_top.animate.set_opacity(0.2),
                x_label_bottom.animate.set_opacity(0.2),
                y_label_top.animate.set_opacity(0.2),
                y_label_bottom.animate.set_opacity(0.2),
                z_label_top.animate.set_opacity(0.2),
                z_label_bottom.animate.set_opacity(0.2),
                )
        text_ac = Text("Example 2: Recall rotation operator" ,font_size=40).fix_in_frame().shift(UP*2)
        self.play(Write(text_ac))
        #equation = Tex(
        #    r"R_z\left(\frac{\pi}{3}\right) \approx S \cdot H \cdot T \cdot S \cdot T \cdot H",
        #    color=WHITE
        #).fix_in_frame().next_to(text_ac, DOWN, aligned_edge=LEFT).shift(LEFT)
        rx_eq = Tex(
            r"R_x(\theta) = \exp(-i(\theta/2)X) = ",
            r"\begin{bmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}",
            r"\quad (X\text{-rotation by }\theta),"
        ).fix_in_frame().scale(0.8)
        
        ry_eq = Tex(
            r"R_y(\theta) = \exp(-i(\theta/2)Y) = ",
            r"\begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}",
            r"\quad (Y\text{-rotation by }\theta),"
        ).fix_in_frame().scale(0.8)
        
        rz_eq = Tex(
            r"R_z(\theta) = \exp(-i(\theta/2)Z) = ",
            r"\begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}",
            r"\quad (Z\text{-rotation by }\theta)."
        ).fix_in_frame().scale(0.8)
        
        # Align equations vertically
        rx_eq.next_to(text_ac, DOWN)
        ry_eq.next_to(rx_eq, DOWN)
        rz_eq.next_to(ry_eq, DOWN)
        
        # Add equations to the scene
        self.play(Write(rx_eq))
        self.play(Write(ry_eq))
        self.play(Write(rz_eq))
        

        #self.play(Write(equation))
        self.wait(1)
        self.play(FadeOut(text_ac),FadeOut(rx_eq),FadeOut(ry_eq),FadeOut(rz_eq))
        eq = Tex(
            f"\\text{{Approximate: }} "
            r"R_z\left(\frac{\pi}{4}\right)",
            color=WHITE
        ).fix_in_frame()
        #self.play(Write(text_ac))
        self.play(Write(eq))
        text_bel = Text("This needs a few more gates...").fix_in_frame().next_to(eq, DOWN, aligned_edge=LEFT).shift(LEFT)
        self.play(Write(text_bel))
        self.wait(1)
        self.play(FadeOut(text_bel),FadeOut(eq))


        self.play(
                        sphere.animate.set_opacity(0.8),
                axes.animate.set_opacity(1),
                x_label_top.animate.set_opacity(1),
                x_label_bottom.animate.set_opacity(1),
                y_label_top.animate.set_opacity(1),
                y_label_bottom.animate.set_opacity(1),
                z_label_top.animate.set_opacity(1),
                z_label_bottom.animate.set_opacity(1),
                )


        # readout
        state_vector = qubit_to_bloch(Rz(np.pi/4)@q_plus)
        vector = Vector(state_vector.squeeze(),color=RED,stroke_width=10)
        self.play(FadeIn(vector))
        #label = Text("target", color=RED,font_size=35).next_to(vector.get_center()*0.8, UP).rotate(vector.get_angle()).shift(LEFT*0.25)
        #self.play(FadeIn(label))
        self.wait(1)

        #------------------------------------
        #shown_images = []
        #H⋅T⋅S⋅T⋅H⋅S

        gates_all = """SHTHTSHTSHTSHTSHTSHTHTHTHTSHTHTHTSHTHTHTHTHTHTSHTSHTSHTHTSHTHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTHTHTSHTHTHTHTSHTHTSHTHTSHTHTSHTHTHTHTSHTHTHTSHTSHTHTSHTHXSSS"""
        str_to_gat = {"S":S,"H":H,"T":T,"X":X}
        gates_toshow = ["q_plus"] + [a for a in gates_all if a in str_to_gat.keys()][::-1]
        im_path , state = image_gate_dict[gates_toshow.pop(0)]
        state_vector = qubit_to_bloch(state)
        vector = Vector(state_vector.squeeze(),color=RED) #TODO tip_shape: "https://docs.manim.community/en/stable/reference/manim.mobject.geometry.tips.ArrowTip.html"
        vector.set_color(RED)
        #image = ImageMobject("gate_imgs/" + im_path)
        #shown_images.append(image)
        #image.scale(0.3).fix_in_frame()
        #image.to_edge(UP).move_to(box.get_center()).shift(RIGHT*4)
        #self.add(vector,image)
        self.play(FadeIn(vector))
        vector_position_text = Tex(
            f"\\text{{Current State: }} "
            f"\\begin{{bmatrix}} {np.round(state[0].item(),2)} \\\\ {np.round(state[1].item(),2)} \\end{{bmatrix}}",
            font_size=40
        ).fix_in_frame().to_corner(UR).shift(LEFT)
        self.play(Write(vector_position_text))
        #t_state = Rx(np.pi/2)@q0
        #print(t_state)
        
        for i, gate_name in enumerate(gates_toshow):
            im_path , gate = image_gate_dict[gate_name]
            state = gate @ state
            #vector_position_text.become(Text(f"Current State: {np.round(state,2)}", font_size=24)).fix_in_frame().to_corner(UR)
            vector_position_text2 = Tex(
            f"\\text{{Current State: }} "
            f"\\begin{{bmatrix}} {np.round(state[0].item(),2)} \\\\ {np.round(state[1].item(),2)} \\end{{bmatrix}}",
            font_size=40
        ).fix_in_frame().to_corner(UR).shift(LEFT)
            
            
            vector_position_text.become(vector_position_text2)
            #self.play(Transform(vector_position_text, vector_position_text2))
            state_vector = qubit_to_bloch(state)
            vector_new = Vector(state_vector.squeeze()).set_color(RED)
            #image = ImageMobject("gate_imgs/" + im_path)
            #shown_images.append(image)
            #image.scale(0.3).fix_in_frame()
            #image.to_edge(UP).move_to(box.get_center()).shift(RIGHT*4).shift(LEFT * ((i+1) / 1))
            self.add(vector)
            self.play(Transform(vector, vector_new))
            

        self.wait(1)
        self.play(FadeOut(vector),FadeOut(vector_position_text),FadeOut(vector_new),*[FadeOut(image) for image in shown_images])
        
        # #self.clear()

        
