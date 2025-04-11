from Biologging_Toolkit.wrapper import Wrapper



class Trajectory(Wrapper) :
    def __init__(self,
                 depid,
                 *,
                 path,
                 ponderation = 'acoustic'
                 ):
        """
        This class uses processed dataset to reconstruct the animal's trajectory.
        The main method is to use Euler angles to get the speed from the pitch and vertical speed.
        If acoustic data is available in the data structure a model can be fitted using the previous speed estimation.
        """

        super().__init__(
            depid,
            path
        )

        self.ponderation = ponderation