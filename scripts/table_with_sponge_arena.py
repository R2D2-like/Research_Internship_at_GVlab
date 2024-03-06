import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion

class TableWithSpongeArena(Arena):
    """
    Workspace that contains a table with a sponge placed at its center.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table.
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table.
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table.
        sponge_size (3-tuple): (L,W,H) dimensions of the sponge.
        has_legs (bool): Whether the table has legs or not.
        xml (str): xml file to load arena.
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        sponge_size=(0.1, 0.1, 0.05),
        sponge_friction=(1, 0.005, 0.0001),
        has_legs=True,
        xml="/root/Research_Internship_at_GVlab/scripts/arenas/table_with_sponge_arena.xml",
    ):
        super().__init__(xml_path_completion(xml))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.sponge_size = np.array(sponge_size)
        self.sponge_friction = sponge_friction
        self.center_pos = self.bottom_pos + np.array([0, 0, -self.table_half_size[2]]) + self.table_offset

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")
        self.sponge_body = self.table_body.find("./body[@name='sponge']")
        self.sponge_collision = self.sponge_body.find("./geom[@name='sponge_collision']")
        self.sponge_visual = self.sponge_body.find("./geom[@name='sponge_visual']")
        # self.sponge_geom = self.sponge_body.find("./geom")

        self.has_legs = has_legs
        self.table_legs_visual = [
            self.table_body.find("./geom[@name='table_leg1_visual']"),
            self.table_body.find("./geom[@name='table_leg2_visual']"),
            self.table_body.find("./geom[@name='table_leg3_visual']"),
            self.table_body.find("./geom[@name='table_leg4_visual']"),
        ]

        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena and places the sponge at the center of the table."""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set("pos", array_to_string(np.array([0, 0, self.table_half_size[2]])))

        # If we're not using legs, set their size to 0
        if not self.has_legs:
            for leg in self.table_legs_visual:
                leg.set("rgba", array_to_string([1, 0, 0, 0]))
                leg.set("size", array_to_string([0.0001, 0.0001]))
        else:
            # Otherwise, set leg locations appropriately
            delta_x = [0.1, -0.1, -0.1, 0.1]
            delta_y = [0.1, 0.1, -0.1, -0.1]
            for leg, dx, dy in zip(self.table_legs_visual, delta_x, delta_y):
                # If x-length of table is less than a certain length, place leg in the middle between ends
                # Otherwise we place it near the edge
                x = 0
                if self.table_half_size[0] > abs(dx * 2.0):
                    x += np.sign(dx) * self.table_half_size[0] - dx
                # Repeat the same process for y
                y = 0
                if self.table_half_size[1] > abs(dy * 2.0):
                    y += np.sign(dy) * self.table_half_size[1] - dy
                # Get z value
                z = (self.table_offset[2] - self.table_half_size[2]) / 2.0
                # Set leg position
                leg.set("pos", array_to_string([x, y, -z]))
                # Set leg size
                leg.set("size", array_to_string([0.025, z]))

        # Place the sponge at the center of the table, on top of it
        sponge_pos = np.array([0, 0, self.table_half_size[2] + self.sponge_size[2] / 2])
        self.sponge_body.set("pos", array_to_string(sponge_pos))
        self.sponge_collision.set("size", array_to_string(self.sponge_size))
        self.sponge_collision.set("friction", array_to_string(self.sponge_friction))
        self.sponge_visual.set("size", array_to_string(self.sponge_size))
        # self.sponge_geom.set("pos", array_to_string(sponge_pos))
        # self.sponge_geom.set("size", array_to_string(self.sponge_size))

    @property
    def table_top_abs(self):
        """
        Grabs the absolute position of table top

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset

    @property
    def sponge_top_abs(self):
        """
        Grabs the absolute position of the sponge top.

        Returns:
            np.array: (x,y,z) sponge position.
        """
        return self.table_top_abs + np.array([0, 0, self.table_half_size[2] + self.sponge_size[2]])

if __name__ == "__main__":
    # Create an instance of the arena
    arena = TableWithSpongeArena()

    # Print out the absolute position of the table top and sponge top
    print("Table top position: ", arena.table_top_abs)
    print("Sponge top position: ", arena.sponge_top_abs)

