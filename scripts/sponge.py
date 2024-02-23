from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, string_to_array

# Define the SpongeObject class
class SpongeObject(MujocoXMLObject):
    def __init__(self, name="sponge", size=None, friction=None, damping=None):
        super().__init__("/root/Research_Internship_at_GVlab/scripts/objects/sponge.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)
        
        # If size is specified, override the default size
        if size is not None:
            self.size = size
            self._set_size(self.size)
        
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_sponge_friction(self.friction)
        if self.damping is not None:
            self._set_sponge_damping(self.damping)

    def _set_size(self, size):
        """
        Helper function to override the sponge size directly in the XML

        Args:
            size (3-tuple of float): size parameters to override the ones specified in the XML
        """
        sponge_body = self.worldbody.find("./body[@name='sponge']")
        sponge_collision = sponge_body.find("./geom[@name='sponge_collision']")
        sponge_geom.set("size", array_to_string(size))

    def _set_sponge_friction(self, friction):
        """
        Helper function to override the sponge friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        sponge = self.worldbody.find("./body")
        sponge_collision = sponge.find("./geom[@name='sponge_collision']")
        sponge_collision.set("friction", array_to_string(friction))

    def _set_sponge_damping(self, damping):
        """
        Helper function to override the sponge friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        sponge = self.worldbody.find("./body")
        sponge.joint.set("damping", str(damping))



