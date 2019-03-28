import dash
import dash_html_components as html
import dash_core_components as dcc


external_stylesheets =  ['/assets/test.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

app.layout = html.Div(children = [html.Div(children =[ html.Ul(children= [
                                        html.Li(html.A('Home',className='active' ,href ='#home')),
                                        html.Li(html.A('News', href ='#news')),
                                        html.Li(html.A('Home', href ='#contact')),
                                        html.Li(html.A('Home', href ='#about')),
                                                     ])],style={'background-image':'url("/assets/diginex_inline_logo.svg")',
                                                                'background-repeat': 'no-repeat','background-position': 'right top',
                                                                'background-size': '300px 30px','height':'5%','position':'fixed',
                                                                'top':'0','border':'3px solid','width':'100%'}),
                                html.Div(style={'padding':'20px','margin-top':'30px','background-color':'#FFFFFF','height':'1500px'},children=[
                                    #html.Div(dcc.Graph(style={'height':'500px'})),
                                    html.H1('Fixed Top Navigation Bar'),
                                    html.H2('Scroll the page to see the effect'),
                                    html.H2('The Navigation bar will stay at the top of the page while scrolling'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                    html.P('Some text some text some text some text..'),
                                ])
                    ])



if __name__ == '__main__':
    app.run_server(debug=True,port =8051)