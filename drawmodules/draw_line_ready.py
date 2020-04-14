import __main__ as m

def call(lp, last_lp):
    print('implemented', lp, last_lp, m)
    if last_lp is not None and 3 < last_lp['length'] < 50 and 3 < lp['length'] < 50 and lp['parallel_to_last'] < 0.3:
        print('kreuz')
        cx = (last_lp['start'].x + last_lp['end'].x + lp['start'].x + lp['end'].x) / 4
        cy = (last_lp['start'].y + last_lp['end'].y + lp['start'].y + lp['end'].y) / 4
        
        m.w.create_line(cx-5, cy-5, cx+5, cy+5, fill="red")
        m.w.create_line(cx-5, cy+5, cx+5, cy-5, fill="red")
        
        # m.w.create_line(last_lp['start'].x,last_lp['start'].y, last_lp['end'].x, last_lp['end'].y, fill="red" )
        # m.w.create_line(lp['start'].x,lp['start'].y, lp['end'].x, lp['end'].y, fill="red" )
        return True
    return False