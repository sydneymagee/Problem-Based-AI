from rgkit import rg

class Robot:
    def act(self, game):
        valid_adj = rg.locs_around(self.location, filter_out=('invalid', 'obstacle'))

        if self.hp <= 5:
            return['suicide']

        # if there are enemies around, attack them
        for loc, bot in game.robots.items():
            if bot.player_id != self.player_id:
                if rg.dist(loc, self.location) <= 1:
                    return ['attack', loc]

        # See if we have an ally near
        #If not, move to closer to an ally
        adj_allies = self.ally_near(game)
        for loc, bot in game.robots.items():
            distance = 10

            if bot.player_id == self.player_id and bot.robot_id != self.robot_id:
                if rg.dist(loc, self.location) < distance and not adj_allies:
                    move_to = rg.toward(self.location, loc)

                    if move_to in valid_adj:
                        return ['move', move_to]


        # Move toward nearest enemy with hp less than or equal to mine
        nearest_enemy = self.nearest_enemy(game)
        if nearest_enemy != ():
            move_to = rg.toward(self.location, nearest_enemy)
            if move_to in valid_adj:
                return ['move', move_to]

        # Otherwise if bot is 2 spaces away, attack towards that direction
        # for loc, bot in game.robots.items():
        #     if bot.player_id != self.player_id:
        #         if rg.dist(loc, self.location) <= 2:
        #             toward = rg.toward(self.location, loc)
        #             return ['attack', toward]

        return['guard']

    def nearest_enemy(self, game):
        smallest_distance = 30
        closest = ()
        for loc, bot in game.robots.items():
            if bot.player_id != self.player_id:
                distance = rg.wdist(self.location, loc)

                if (distance < smallest_distance and bot.hp <= self.hp):
                    closest = loc
        return closest

    def ally_near(self, game):
        for loc, bot in game.robots.items():
            if bot.player_id == self.player_id and bot.robot_id != self.robot_id:
                dist = rg.wdist(loc, self.location)
                if dist <= 3:
                    return True
        return False

    def valid_move(self, loc):
        locations = rg.loc_types(loc)
        if 'invalid' in locations or 'obstacle' in locations or 'spawn' in locations:
            return False
        return True