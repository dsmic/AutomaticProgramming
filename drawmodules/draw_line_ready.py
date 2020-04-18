# pylint: disable=C0301, C0114, C0103, C0116, R0912, R0915, R0914, R0911
import __main__ as m
def call(lp, last_lp):
    print('implemented', lp, last_lp, m)
    flatpoint = [p for s in lp['pointlist'] for p in (s.x, s.y)]

    # check for click
    if lp['length'] < 5:

        # set mark or click position
        if m.mark is None:
            np = m.find_point_near(lp['start'], 20)
            if np is None:
                intersecs = m.find_intersections()
                print('intersecs', intersecs)
                mindist = None
                minl = None
                for l in intersecs:
                    print('p', l.x, l.y)
                    dist = m.abst(lp['start'], l)
                    if mindist is None or dist < mindist:
                        mindist = dist
                        minl = l
                if mindist is not None and mindist < 30:
                    m.draw_objects.append(m.draw_point(minl.x, minl.y, 'blue'))
                    #m.mark = None
                else:
                    # this allows to draw points by just clicking, not making a cross
                    m.draw_objects.append(m.draw_point(lp['start'].x, lp['start'].y))
            else:
                print('mark set')
                m.mark = m.draw_mark(np)
            return True

        # create a circle with the same radius
        if m.mark is not None:
            if m.abst(m.mark, lp['start']) < 30:
                m.draw_objects.append(m.draw_circle(m.mark, m.lastradius))

        m.mark = None
        return True


    # check for point was marked, now we make a circle from it
    if m.mark is not None:
        # check if point is near center
        pp = m.find_point_near(lp['center'], 20)
        if pp is None:
            pp = lp['center']
        radius = m.abst(m.mark, pp)
        m.lastradius = radius
        m.draw_objects.append(m.draw_circle(m.mark, radius))
        m.mark = None
        return True

    # check for a cross to mark points
    if last_lp is not None and 5 < last_lp['length'] < 80 and 5 < lp['length'] < 80 and lp['parallel_to_last'] < 0.3 and m.abst(lp['center'], last_lp['center']) < 20:
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

    # check if we do lengthen a line
    nl_list = m.find_line_near(lp['start'], 20)
    if len(nl_list) > 0:
        print('nl', nl_list)
        for nl in nl_list:
            # check for direction
            line_direct = m.direct(nl.sp, nl.ep)
            is_par = m.is_parallel(line_direct, lp['direction'])
            if is_par > 0.8:
                print('same direction')
                if m.abst(nl.sp, lp['start']) < 20:
                    if not nl.sg:
                        m.draw_objects.append(m.changed_line(nl, nl.sg, nl.eg))
                        nl.sg = True
                if m.abst(nl.ep, lp['start']) < 20:
                    if not nl.eg:
                        m.draw_objects.append(m.changed_line(nl, nl.sg, nl.eg))
                        nl.eg = True

        return True


    # The hand drawn element is saved
    m.draw_objects.append(m.draw_polygon(flatpoint))
    return False
