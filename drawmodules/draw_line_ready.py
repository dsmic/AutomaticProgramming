import __main__ as m

def call(lp, last_lp):
    print('implemented', lp, last_lp, m)
    flatpoint = [p for s in lp['pointlist'] for p in (s.x,s.y)]
    
    # check for click
    if lp['length'] == 0:
        if m.markedpoint is None and m.clickedposition is None:
            np = m.find_point_near(lp['start'], 20)
            m.markedpoint = np
            if np is None:
                m.clickedposition = lp['start']
            else:
                m.clickedposition = None
            return True
        intersecs = m.find_intersections()
        print('intersecs', intersecs)
        for l in intersecs:
            print('p', l.x,l.y)
            m.draw_objects.append(m.draw_point(l.x,l.y, 'blue'))
        m.markedpoint = None
        m.clickedposition = None
        return True
    
    
    # check for point was marked, now we make a circle from it
    if m.markedpoint is not None:
        # check if point is near center
        pp = m.find_point_near(lp['center'],20)
        if pp is None:
            pp = lp['center']
        radius = m.abst(m.markedpoint, pp)
        m.draw_objects.append(m.draw_circle(m.markedpoint, radius))
        m.markedpoint = None
        return True
        
    # check for a cross to mark points    
    if last_lp is not None and 3 < last_lp['length'] < 50 and 3 < lp['length'] < 50 and lp['parallel_to_last'] < 0.3:
        cx = (last_lp['start'].x + last_lp['end'].x + lp['start'].x + lp['end'].x) / 4
        cy = (last_lp['start'].y + last_lp['end'].y + lp['start'].y + lp['end'].y) / 4
        m.draw_objects.pop()
        m.draw_objects.append(m.draw_point(cx, cy))
        return True
    
    # check for a line between points
    sp = m.find_point_near(lp['start'], 20)
    ep = m.find_point_near(lp['end'], 20)
    if sp is not None and ep is not None:
        m.draw_objects.append(m.draw_line(sp,ep))
        return True

    # The drawn element is saved
    m.draw_objects.append(m.draw_polygon(flatpoint))
    return False