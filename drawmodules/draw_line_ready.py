# pylint: disable=C0301, C0114, C0103, C0116, R0912
import __main__ as m
def call(lp, last_lp):
    print('implemented', lp, last_lp, m)
    flatpoint = [p for s in lp['pointlist'] for p in (s.x, s.y)]

    # check for click
    if lp['length'] < 3:
        if m.clickedposition is not None:
            if m.abst(m.clickedposition, lp['start']) < 30:
                intersecs = m.find_intersections()
                print('intersecs', intersecs)
                mindist = None
                minl = None
                for l in intersecs:
                    print('p', l.x, l.y)
                    dist = m.abst(m.clickedposition, l)
                    if mindist is None or dist < mindist:
                        mindist = dist
                        minl = l
                if mindist is not None and mindist < 30:
                    m.draw_objects.append(m.draw_point(minl.x, minl.y, 'blue'))

        if m.markedpoint is None and m.clickedposition is None:
            np = m.find_point_near(lp['start'], 20)
            m.markedpoint = np
            if np is None:
                m.clickedposition = lp['start']
                print('clickedposition set')
            else:
                print('markedpoint set')
                m.mark = m.draw_mark(m.markedpoint)
            return True # markedpoints or clickpositon set

        m.markedpoint = None
        m.mark = None

        m.clickedposition = None
        return True

    m.clickedposition = None

    # check for point was marked, now we make a circle from it
    if m.markedpoint is not None:
        # check if point is near center
        pp = m.find_point_near(lp['center'], 20)
        if pp is None:
            pp = lp['center']
        radius = m.abst(m.markedpoint, pp)
        m.draw_objects.append(m.draw_circle(m.markedpoint, radius))
        m.markedpoint = None
        m.mark = None
        return True

    # check for a cross to mark points
    if last_lp is not None and 3 < last_lp['length'] < 50 and 3 < lp['length'] < 50 and lp['parallel_to_last'] < 0.3 and m.abst(lp['center'], last_lp['center']) < 20:
        cx = (last_lp['start'].x + last_lp['end'].x + lp['start'].x + lp['end'].x) / 4
        cy = (last_lp['start'].y + last_lp['end'].y + lp['start'].y + lp['end'].y) / 4
        m.draw_objects.pop()
        m.draw_objects.append(m.draw_point(cx, cy))
        return True

    # check for a line between points
    sp = m.find_point_near(lp['start'], 20)
    ep = m.find_point_near(lp['end'], 20)
    if sp is not None and ep is not None:
        m.draw_objects.append(m.draw_line(sp, ep))
        return True

    # The drawn element is saved
    m.draw_objects.append(m.draw_polygon(flatpoint))
    return False
