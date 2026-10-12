import unittest
import numpy as np
from endpoint_policy import resolve_endpoints


class EndpointPolicyTest(unittest.TestCase):
    def test_exact_hand_tokens_arm_ancestry_pair_and_wrist_chain(self):
        names = ['Hips', 'LeftShoulder', 'LeftForeArm', 'LeftHand',
                 'RightShoulder', 'RightForeArm', 'RightWrist', 'RightIndex1']
        parents = [-1, 0, 1, 2, 0, 4, 5, 6]
        offsets = np.zeros((len(names), 3)); offsets[1:, 0] = .1
        result = resolve_endpoints('mixamo', 'unimate', names, parents, offsets)
        self.assertEqual(result['indices'], [3, 6])
        self.assertEqual(result['strategy'], 'double_hand_or_wrist_anchors')
        self.assertEqual(result['names'], ['LeftHand', 'RightWrist'])
        self.assertTrue(result['valid'])

    def test_nonhand_domain_allowlist_and_substring_rejection(self):
        names = ['Root', 'LeftArm', 'LeftHandle', 'RightArm', 'RightHandrail',
                 'LeftFootHand', 'RightWingHand']
        parents = [-1, 0, 1, 0, 3, 1, 3]
        offsets = np.zeros((len(names), 3)); offsets[1:, 0] = .1
        result = resolve_endpoints('mixamo', 'unimate', names, parents, offsets)
        self.assertEqual(result['kind'], 'generic_topological_leaf')
        self.assertEqual(result['indices'], [2, 4, 5, 6])
        self.assertEqual(len(result['hand_candidates_rejected']), 2)
        # Real-looking names cannot enable hand inference for quadrupeds or XWBC.
        for body, domain in [('79fed889082e4a25ad92be75b57f641c', 'unimate'), ('mixamo', 'xwbc')]:
            result = resolve_endpoints(body, domain, ['Root', 'LeftArm', 'LeftHand'], [-1, 0, 1], [[0,0,0],[1,0,0],[.1,0,0]])
            self.assertEqual(result['kind'], 'generic_topological_leaf')
            self.assertFalse(result['hand_resolution_enabled'])

    def test_zero_length_terminal_kept_using_cumulative_path(self):
        result = resolve_endpoints('snake', 'unimate', ['Root', 'Segment', 'Tip'], [-1,0,1], [[0,0,0],[1,0,0],[0,0,0]])
        self.assertEqual(result['indices'], [2])
        self.assertEqual(result['cumulative_rest_path_lengths'][2], 1.)
        result = resolve_endpoints('empty', 'unimate', ['Root', 'Tip'], [-1,0], [[0,0,0],[0,0,0]])
        self.assertFalse(result['valid'])
        self.assertTrue(result['isolate_endpoint_condition'])
        self.assertEqual(result['indices'], [])

    def test_ambiguous_hands_missing_chain_and_invalid_tree_fail_closed(self):
        names = ['Root', 'LeftArm', 'LeftHand', 'LeftHand', 'RightWrist']
        parents = [-1,0,1,1,0]
        offsets = np.zeros((len(names),3)); offsets[1:,0] = .1
        result = resolve_endpoints('mixamo', 'unimate', names, parents, offsets)
        self.assertEqual(result['kind'], 'generic_topological_leaf')
        self.assertEqual(result['ambiguous_hand_sides'], ['left'])
        reasons = [reason for item in result['hand_candidates_rejected'] for reason in item['reasons']]
        self.assertIn('missing_arm_chain_ancestor', reasons)
        self.assertIn('wrist_without_hand_or_finger_descendant', reasons)
        result = resolve_endpoints('mixamo', 'unimate', ['Root','LeftHand'], [-1,1], [[0,0,0],[1,0,0]])
        self.assertFalse(result['valid'])
        self.assertEqual(result['indices'], [])


    def test_xwbc_distal_wrist_only_chain_and_named_hand_preference(self):
        names = ['Root', 'left_shoulder_link', 'left_wrist_roll_link',
                 'left_wrist_pitch_link', 'left_wrist_yaw_link',
                 'right_shoulder_link', 'right_wrist_roll_link',
                 'right_wrist_pitch_link', 'right_wrist_yaw_link']
        parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7]
        offsets = np.zeros((len(names), 3)); offsets[[1, 2, 5, 6], 0] = .2
        result = resolve_endpoints('unitree_g1_29dof', 'xwbc', names, parents, offsets)
        self.assertEqual(result['indices'], [4, 8])
        self.assertEqual(result['strategy'], 'double_wrist_anchors')
        self.assertEqual([item['label'] for item in result['endpoint_labels']], ['named_wrist_kinematic_anchor'] * 2)
        self.assertTrue(all(item['all_wrist_candidates_form_ancestor_chain'] for item in result['wrist_chain_selection_audit']))
        # Zero local offsets at distal axes still have a valid cumulative path.
        self.assertGreater(result['cumulative_rest_path_lengths'][4], 0)
        names.append('left_rubber_hand'); parents.append(4); offsets = np.vstack([offsets, [0, 0, .1]])
        result = resolve_endpoints('unitree_g1_29dof', 'xwbc', names, parents, offsets)
        self.assertEqual(result['indices'], [9, 8])
        self.assertEqual(result['endpoint_labels'][0]['label'], 'named_hand_kinematic_anchor')

    def test_xwbc_wrist_forks_unknown_sides_and_excluded_bodies_stay_generic(self):
        names = ['Root', 'LeftShoulder', 'LeftWristRoll', 'LeftWristA', 'LeftWristB']
        parents = [-1, 0, 1, 2, 2]
        offsets = np.zeros((len(names), 3)); offsets[1:, 0] = .1
        result = resolve_endpoints('unitree_h1_2', 'xwbc', names, parents, offsets)
        self.assertEqual(result['kind'], 'generic_topological_leaf')
        self.assertEqual(result['indices'], [3, 4])
        self.assertEqual(result['ambiguous_hand_sides'], ['left'])
        result = resolve_endpoints('human_smpl', 'xwbc', ['Root', 'Shoulder', 'Wrist'], [-1, 0, 1], [[0,0,0],[1,0,0],[1,0,0]])
        self.assertEqual(result['kind'], 'generic_topological_leaf')
        for body in ['pal_talos', 'engineai_pm01', 'soma77', 'g1_30', 'unverified_robot']:
            result = resolve_endpoints(body, 'xwbc', ['Root', 'LeftShoulder', 'LeftHand'], [-1,0,1], [[0,0,0],[1,0,0],[1,0,0]])
            self.assertFalse(result['hand_resolution_enabled'])
            self.assertEqual(result['kind'], 'generic_topological_leaf')


if __name__ == '__main__':
    unittest.main()
