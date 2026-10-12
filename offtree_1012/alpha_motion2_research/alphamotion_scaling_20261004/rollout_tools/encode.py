import json,subprocess
from pathlib import Path
import numpy as np

def encode(frames,times,output):
    """VFR at a microsecond timebase; final sample held for median recorded interval."""
    times=np.asarray(times);dt=np.diff(times);hold=float(np.median(dt));timeline=times-times[0]
    # Explicit PTS expression supports the host's older FFmpeg without concat options.
    pts_expression='+'.join(f'eq(N\\,{i})*{round(float(v)*1000000)}' for i,v in enumerate(timeline))
    subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(1/hold),'-i',str(Path(frames)/'%06d.png'),'-vf','settb=1/1000000,setpts='+pts_expression,'-vsync','vfr','-enc_time_base','1:1000000','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-video_track_timescale','1000000','-movflags','+faststart',str(output)],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-'],check=True)
    d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_frames','-show_format','-of','json',str(output)]))
    pts=np.array([float(x['best_effort_timestamp_time']) for x in d['frames'] if x['media_type']=='video']);expected=timeline
    if len(pts)!=len(expected) or np.max(np.abs(pts-expected))>4e-6:raise ValueError('encoder changed frame identity/time: '+str(pts))
    stream=d['streams'][0]
    if stream['codec_name']!='h264' or stream['pix_fmt']!='yuv420p' or (stream['width'],stream['height'])!=(1920,1080):raise ValueError('web video encoding mismatch')
    b=Path(output).read_bytes()
    if b.find(b'moov')>b.find(b'mdat'):raise ValueError('faststart absent')
    return {'source_frames':len(times),'encoded_frames':len(pts),'end_hold_seconds':hold,'terminal_duplicate':'none; final source sample held for median recorded interval','source_timestamp_span_seconds':float(timeline[-1]),'duration_seconds':float(d['format']['duration']),'max_pts_error_seconds':float(np.max(np.abs(pts-expected))),'avg_frame_rate':stream['avg_frame_rate'],'r_frame_rate':stream['r_frame_rate'],'codec':'h264','pixel_format':'yuv420p','faststart':True,'full_decode':'passed','frame_pts_seconds':pts.tolist()}
