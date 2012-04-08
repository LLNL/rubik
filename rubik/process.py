

class Process(object):
    """The process class represents a single task in a parallel application with a
       unique identifier.  Identifiers can be anything.
    """
    def __init__(self, id):
        """Constructs a process with a particular id, optionally as part of a list.
           Parameters:
             id      arbitrary process identifier.
        """
        self.id      = id
        self.coord   = None

    def __str__(self):
        """String representation for printing is just the identifier."""
        return "<Process %d>" % self.id
