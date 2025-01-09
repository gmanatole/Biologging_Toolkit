import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def subplots_centered(nrows, ncols, figsize, nfigs, sharex=False, sharey=False):
    """
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
        sharex: bool, optional, default: False
            If True, the x-axis will be shared among subplots.
        sharey: bool, optional, default: False
            If True, the y-axis will be shared among subplots.
    """
    assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
    
    fig = plt.figure(figsize=figsize)
    axs = []
    
    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    shared_x = None  # For handling shared x-axis
    shared_y = None  # For handling shared y-axis

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1:  # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0

        # Create subplot
        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off], 
                          sharex=shared_x if sharex else None,
                          sharey=shared_y if sharey else None)
        
        # Set the first axes as the reference for sharing
        if shared_x is None and sharex:
            shared_x = ax
        if shared_y is None and sharey:
            shared_y = ax

        axs.append(ax)
    
    # Adjust layout for shared axes
    if sharex:
        for ax in axs:
            if ax != shared_x:
                ax.sharex(shared_x)
    if sharey:
        for ax in axs:
            if ax != shared_y:
                ax.sharey(shared_y)
        
    return fig, axs