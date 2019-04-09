from bokeh.plotting import figure

def initialize_figure(title, length):
    if title is None:
        title = 'Untitled'

    tooltips = [
        ("x", "$x"),
        ("y", "$y"),
        ('dist', '@image'),
    ]
    return figure(x_range=(0, length), y_range=(0, length), tooltips=tooltips)

def draw_pairwise_distance(pdist, p=None, x=None, y=None, dw=None, dh=None, title=None, palette="Greys256"):
    """
    Recommended palette:
        ['Inferno256', 'Greys256', 'Cividis256']
    """
    if p is None:
        p = initialize_figure(title, pdist.shape[0])

    if x is None:
        x = 0
        y = 0
        dw = pdist.shape[0]
        dh = pdist.shape[1]
    p.image([pdist], x=x, y=y, dw=dw, dh=dh, palette=palette)
    return p
