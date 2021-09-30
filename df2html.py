import pandas as pd

__all__ = ['df2html']

def df2html(df):
    """
    From https://stackoverflow.com/a/49687866/2007153
    From https://stackoverflow.com/questions/52104682/rendering-a-pandas-dataframe-as-html-with-same-styling-as-jupyter-notebook
    Get a Jupyter like html of pandas dataframe"""

    styles = [
        dict(selector=" ", 
             props=[("margin","0"),
                    ("font-family",'"Helvetica", "Arial", sans-serif'),
                    ("border-collapse", "collapse"),
                    ("border","none"),
    #               ("border", "2px solid #ccf")
                       ]),

    #header color - optional
    #     dict(selector="thead", 
    #          props=[("background-color","#cc8484")
    #                ]),

        #background shading
        dict(selector="tbody tr:nth-child(even)",
             props=[("background-color", "#fff")]),
        dict(selector="tbody tr:nth-child(odd)",
             props=[("background-color", "#eee")]),

        #cell spacing
        dict(selector="td", 
             props=[("padding", ".5em")]),

        #header cell properties
        dict(selector="th", 
             props=[("font-size", "100%"),
                    ("text-align", "center")]),
    ]
    return df.style.set_table_styles(styles).render()