from manim import *

class ExampleSphere(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PI / 6, theta=PI / 6)
        sphere = Sphere(
            center=(3, 0, 0),
            radius=1,
            resolution=(20, 20),
            u_range=[0.001, PI - 0.001],
            v_range=[0, TAU]
        )
        sphere.set_color(RED)
        vector = Vector([0,0,1])
        self.add(sphere,vector)
        #self.play()

class SpiralInExample2(Scene):
    def construct(self):
        shapes = VGroup(ExampleSphere())
        self.play(SpiralIn(shapes))


class SpiralInExample(Scene):
    def construct(self):
        pi = MathTex(r"\pi").scale(7)
        pi.shift(2.25 * LEFT + 1.5 * UP)
        circle = Circle(color=GREEN_C, fill_opacity=1).shift(LEFT)
        square = Square(color=BLUE_D, fill_opacity=1).shift(UP)
        shapes = VGroup(pi, circle, square)
        self.play(SpiralIn(shapes))
#%manim -qm -v WARNING ExampleSphere