from perceptual_networks import PerceptualNetwork

pn = PerceptualNetwork("VGG19", 6, (40, 40, 3))
pn.summary()
