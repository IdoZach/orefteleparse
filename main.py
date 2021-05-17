from telethon import TelegramClient, events, sync
from pathlib import Path
from datetime import datetime as date
import datetime, re, numpy as np
import pandas as pd
import time
from geopy.geocoders import Nominatim
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, DateSlider, Slider, Label, CustomJS, LabelSet, Button
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from pyproj import Proj, transform, Transformer
from creds import my_creds # dictionary with api_id and api_hash keys

longlat2mercator = Transformer.from_proj("epsg:4326", "epsg:3857")
good_etas = 'דקה', 'דקה וחצי', '45 שניות', '15 שניות', '30 שניות'

def get_client():
    # get from https://my.telegram.org
    api_id =  my_creds['api_id']
    api_hash = my_creds['api_hash']
    client = TelegramClient('session_name', api_id, api_hash)
    client.start()
    return client

def get_messages(client, username='PikudHaOref_all'):
    text = []
    async def main():
        async for message in client.iter_messages(username):
            text.append(message.text)
    with client:
        client.loop.run_until_complete(main())
    return text

def save_messages(text,txt_fname:Path=Path('/tmp/oref_telegram.txt')):
    text = [x for x in text if x is not None]
    txt_fname.write_text('\n'.join(text))

def parse_timeline(fname:Path=Path('/tmp/oref_telegram.txt'), show_unrecognized=False):
    text = fname.read_text()
    rows = text.split('\n')[::-1]
    cur_time, cur_area = None, None
    ret = []
    for l in rows:
        # first: time format ([11/5/2021] 13:13)
        cur_time0 = re.findall(r'\[.*\/.*\/.*\] \d+:\d+',l)
        cur_time1 = re.findall(r'\[.*\/.*\/.*\].* \d+:\d+ ',l)
        if len(cur_time0):
            cur_time = datetime.datetime.strptime(cur_time0[0],'[%d/%m/%Y] %H:%M')
            # print(cur_time[0], cur_time)
        elif len(cur_time1): # alt
            parts = cur_time1[0].split()
            cur_time = []
            for p in parts:
                if '/' in p:
                    cur_time.append(p.strip())
                elif ':' in p:
                    cur_time.append(p.strip())
            cur_time = datetime.datetime.strptime(' '.join(cur_time), '[%d/%m/%Y] %H:%M')
        else:
            # second: area format: ** ... **
            s = 'אזור'
            cur_area0 = re.findall(f'\*\*{s} (.*)\*\*', l)
            if len(cur_area0):
                cur_area = cur_area0[0].strip()
            else:
                # third: city: city name (time to take cover)
                cur_loc = re.findall(f'(.*) \((.*)\)',l)
                if len(cur_loc):
                    loc, eta = cur_loc[0]
                    ret.append(dict(dt=cur_time,area=cur_area,location=loc,eta=eta))
                elif len(l.strip()) and show_unrecognized:
                    print('unrecognized row:')
                    print(l)
    df = pd.DataFrame(ret)
    df['date'] = pd.to_datetime(df['dt'])
    df = df.drop('dt',axis=1)
    return df

def process_latlong(df, out_fname:Path):
    df = df.loc[df['eta'].isin(good_etas)]
    geolocator = Nominatim(user_agent="my_app")
    latlongs={}
    def get_latlong(s):
        if s not in latlongs:
            try:
                location = geolocator.geocode(s)
                latlongs[s] = location.latitude, location.longitude
            except:
                print(f'error in {location}')
                latlongs[s] = -1,-1
        return pd.Series(latlongs[s])
    df[['lat','long']] = df['location'].apply(get_latlong)
    if out_fname is not None:
        df.to_csv(out_fname)
    return df

def eta2color(s):
    # good_etas = 'דקה', 'דקה וחצי', '45 שניות', '15 שניות', '30 שניות'
    colors = ['sienna','maroon','peru','rosybrown','sandybrown'][::-1]
    m = dict(zip(good_etas,colors))
    return m[s]

def gen_bokeh(df,
                il_center = (3879662.3653308493, 3694724.561061665),
                marg_x = 2.0e5, marg_y = 2.0e5,
                init_date = datetime.datetime(year=2021,month=5,day=10),
                MAX_W=1000,MAX_H=800,
             ):

    # main figure with world map
    tile_provider = get_provider(CARTODBPOSITRON)
    p = figure(x_range=(il_center[0]-marg_x,il_center[0]+marg_x),
               y_range=(il_center[1]-marg_y,il_center[1]+marg_y),
               x_axis_type="mercator", y_axis_type="mercator")
    p.plot_height=MAX_H//2
    p.plot_width=MAX_W
    p.add_tile(tile_provider)

    # process df for plotting
    df['mydate'] = pd.to_datetime(df['date'])
    df = df.loc[df['mydate'] > init_date]
    merc = longlat2mercator.transform(df['lat'].values,df['long'].values)
    df['x']=merc[0]
    df['y']=merc[1]
    df['name'] = df['location']
    df[['x','y','mydate_f','name']] = df[['x','y','mydate','name']].fillna('')
    df['fill_alpha'] = .5
    df['line_alpha'] = .0
    df['fill_color'] = df['eta'].apply(eta2color)
    src_cols = ['x','y','mydate_f','name','fill_alpha','line_alpha','fill_color']
    source0 = df[src_cols].to_dict('list')
    source = ColumnDataSource(data=source0)
    # add alarms on map
    p.circle(x="x", y="y", size=4, fill_color="fill_color", fill_alpha='fill_alpha', line_alpha='line_alpha', source=source)
    p.add_tools(HoverTool(
        tooltips=[
            ('time','@mydate_f{%H:%M}'),
            ('date','@mydate_f{%d/%m/%Y}'),
            ('name','@name'),
        ],
        formatters={ '@mydate_f': 'datetime' },
    ))

    # build other elements
    date_slider = DateSlider(start=df['mydate'].min(), end=date.today(), step=24,
             value=date.today(), title='Date', width=int(MAX_W*.9))
    dur_slider = Slider(start=0, end=168, step=1,
                        value=24, title='Hour duration', width=MAX_W//2)
    delay_slider = Slider(start=0, end=1000, step=10,
                        value=200, title='Animation delay (ms)', width=MAX_W//2)

    label_xy = longlat2mercator.transform(30.5,34)
    label_src = ColumnDataSource(data=dict(text=['Hello, use the sliders']))
    label = LabelSet(text='text',x=label_xy[0],y=label_xy[1],source=label_src)
    p.add_layout(label)
    button = Button(label="Play", button_type="success",width=MAX_W//10)

    # add plot for number of alarms per day
    ff = figure(x_range=(df['mydate_f'].min(),df['mydate_f'].max()),x_axis_type="datetime")
    df['day'] = df['mydate_f'].dt.day
    agg = df.groupby('day').agg('count')
    agg['date'] = agg.index.to_series().apply(lambda x: df.loc[df['mydate_f'].dt.day==x,'mydate_f'].iloc[0])
    ff.scatter(agg['date'],agg['Unnamed: 0'],size=10)
    ff.plot_height=MAX_H//3
    ff.plot_width=MAX_W
    ff.title='alarms per day'

    # callback functions
    callback_id = None
    def update_map_callback(attr, old, new):
        st, en = date_slider.value, date_slider.value+dur_slider.value*60*60*1000
        st = datetime.datetime.fromtimestamp(st/1000.)
        en = datetime.datetime.fromtimestamp(en/1000.)
        ddf = df.loc[(df['mydate_f']>=st) & (df['mydate_f']<en)]
        source.data = ddf[src_cols].to_dict('list')
        sts = st.strftime("%d/%m, %H:%M")
        ens = en.strftime("%d/%m, %H:%M")
        label_src.data = dict(text=[f'Range: {sts} to {ens}, total: {len(ddf)}'])
    def restart_animation(attr,old,new):
        global callback_id
        if button.label == 'Stop':
            curdoc().remove_periodic_callback(callback_id)
            callback_id = curdoc().add_periodic_callback(adv_slider, delay_slider.value)
    def adv_slider():
        date_slider.value += 1*1000*60*60
        if date_slider.value >= date_slider.end:
            date_slider.value = date_slider.start
    def button_callback():
        global callback_id
        if button.label == 'Play':
            button.label = 'Stop'
            callback_id = curdoc().add_periodic_callback(adv_slider, delay_slider.value)
        else:
            button.label = 'Play'
            curdoc().remove_periodic_callback(callback_id)

    # callback assignment
    delay_slider.on_change('value',restart_animation)
    date_slider.on_change('value',update_map_callback)
    dur_slider.on_change('value',update_map_callback)
    button.on_click(button_callback)

    lay = column(p, row(date_slider,button), row(dur_slider,delay_slider),ff)
    return lay

if __name__ == '__main__':
    # prepare data
    client = get_client()
    messages = get_messages(client)
    save_messages(messages)
    df = parse_timeline()
    process_latlong(df,out_fname=Path('/tmp/oref_proc.csv'))

if __name__ != '__main__': # server: bokeh server --show main.py
    print(__name__)
    df = pd.read_csv('/tmp/oref_proc.csv') # gen this through main scope with telegram credentials
    # jitter
    df['lat'] += np.random.randn(len(df['lat']))*0.003
    df['long'] += np.random.randn(len(df['long']))*0.003
    ret = gen_bokeh(df)
    curdoc().add_root(ret)