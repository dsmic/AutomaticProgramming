import __main__ as m

def call(lp, last_lp):
    print('implemented', lp, last_lp, m)
    flatpoint = [p for s in lp['pointlist'] for p in (s.x,s.y)]
    print(flatpoint)
    if m.markedpoint is not None:
        print('after marked')
        radius = m.abst(m.markedpoint, lp['center'])
        m.draw_objects.append(m.draw_circle(m.markedpoint, radius))
        m.markedpoint = None
        m.w.delete("all")
        m.lastpoints = None
        return True
        
    if lp['length'] == 0:
        np = m.find_point_near(lp['start'], 20)
        m.markedpoint = np
        print('only clicked', np)
        
    if last_lp is not None and 3 < last_lp['length'] < 50 and 3 < lp['length'] < 50 and lp['parallel_to_last'] < 0.3:
        print('kreuz')
        cx = (last_lp['start'].x + last_lp['end'].x + lp['start'].x + lp['end'].x) / 4
        cy = (last_lp['start'].y + last_lp['end'].y + lp['start'].y + lp['end'].y) / 4
        
        #m.w.create_line(cx-5, cy-5, cx+5, cy+5, fill="red")
        #m.w.create_line(cx-5, cy+5, cx+5, cy-5, fill="red")
        m.draw_objects.pop()
        m.draw_objects.append(m.draw_point(cx, cy))
        m.w.delete("all")
        m.lastpoints = None
        return True
    sp = m.find_point_near(lp['start'], 20)
    ep = m.find_point_near(lp['end'], 20)
    if sp is not None and ep is not None:
        m.draw_objects.append(m.draw_line(sp,ep))
        m.w.delete("all")
        m.lastpoints = None
        return True
    m.draw_objects.append(m.draw_polygon(flatpoint))
    return False