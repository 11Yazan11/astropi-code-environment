# can't run the code on VsCode, need to run it on the simulator because picamzero is unknown.

from picamzero import Camera
from astro_pi_orbit import ISS

iss = ISS()
camera = Camera()

def take_photo(image_name):
    """
    Takes a photo and embeds the current coordinates of the ISS
    into the metadata.
    """
    point = iss.coordinates()
    coordinates = (point.latitude.signed_dms(), point.longitude.signed_dms())
    cam.take_photo(image_name, gps_coordinates=get_gps_coordinates(iss))

take_photo("tagged-img.jpg") 

# This can be quite useful, a lot actually.
# Because we can also put the ISS coordinates in the meta data of each image.
# So we average the speed found from what the images show and from how much the ISS actually moved between those images.
# But in the mean time, I think "how much the ISS actually moved between those images" is easily found with the coordinates and much much more precise.
# So why keep the other method? Ask GPT.