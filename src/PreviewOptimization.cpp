#include <dwl/ocp/PreviewOptimization.h>


namespace dwl
{

namespace ocp
{

PreviewOptimization::PreviewOptimization() : warm_point_(false),
		support_margin_(0.), desired_yaw_B_(0.), cot_weight_(0.),
		num_feet_(0), num_steps_(0), num_stances_(0), num_controls_(1),
		init_schedule_(false), is_bound_(false), collect_data_(true)
{
	unsigned int cycles = 1;
	for (unsigned int i = 0; i < cycles; ++i) {
		schedule_.addPhase(simulation::PreviewPhase(simulation::STANCE));
		schedule_.addPhase(
				simulation::PreviewPhase(simulation::STANCE,
										 dwl::model::ElementList(1,"lh_foot")));
		schedule_.addPhase(
				simulation::PreviewPhase(simulation::STANCE,
										 dwl::model::ElementList(1,"lf_foot")));

		schedule_.addPhase(simulation::PreviewPhase(simulation::STANCE));
		schedule_.addPhase(
				simulation::PreviewPhase(simulation::STANCE,
										 dwl::model::ElementList(1,"rh_foot")));
		schedule_.addPhase(
				simulation::PreviewPhase(simulation::STANCE,
										 dwl::model::ElementList(1,"rf_foot")));
	}

	// Getting the number of phases
	phases_ = schedule_.getNumberPhases();

	// Getting the number of steps
	for (unsigned int k = 0; k < phases_; ++k) {
		simulation::PreviewPhase phase = schedule_.getPhase(k);

		if (phase.doStep())
			num_steps_ += 1;
		else
			num_stances_ += 1;
	}

	set_schedule_ = true;
}


PreviewOptimization::~PreviewOptimization()
{
	cdata_.stopCollectData();
}


void PreviewOptimization::init(bool only_soft_constraints)
{
	// Getting the number of feet and their names
	num_feet_ =	preview_.getFloatingBaseSystem()->getNumberOfEndEffectors(model::FOOT);
	feet_ = preview_.getFloatingBaseSystem()->getEndEffectorList(model::FOOT);
	schedule_.setFeet(feet_);

	// Getting the dimension of the decision vector
	state_dimension_ = getControlDimension();
	constraint_dimension_ = 0;

	if (collect_data_) {
		utils::CollectData::Tags data_tags{"cmd_cost",
										   "energy_cost",
										   "cop_cost",
										   "prev_cost",
										   "terrain_cost",
										   "total_cost"};
		for (unsigned int n = 0; n < getControlDimension(); n++)
			data_tags.push_back("x_" + std::to_string(n));
		cdata_.initCollectData("preview_opt.dat", data_tags);
	}
}


void PreviewOptimization::setActualWholeBodyState(const WholeBodyState& state)
{
	// Converting the full to reduced-body state
	ReducedBodyState reduced_state;
	preview_.fromWholeBodyState(reduced_state, state);

	// Setting up the actual reduced-body state
	setActualReducedBodyState(reduced_state);
}


void PreviewOptimization::setActualReducedBodyState(const ReducedBodyState& state)
{
	actual_state_ = state;

	// Getting the index of the actual phase
	if (!init_schedule_) {
		schedule_.init(actual_state_.support_region);
		init_schedule_ = true;
	}

	// Getting the actual phase mapping
	phase_id_.resize(phases_);
	unsigned int phase_idx = schedule_.getIndex();
	for (unsigned int k = 0; k < phases_; ++k)
		phase_id_[k] = (k + phases_ + phase_idx) % phases_;
}


void PreviewOptimization::setStartingPreviewControl(const simulation::PreviewControl& control)
{
	// Setting the starting preview control
	warm_point_ = true;
	warm_control_ = control;
}


void PreviewOptimization::setBounds(PreviewBounds bounds)
{
	bounds_ = bounds;
	is_bound_ = true;
}


void PreviewOptimization::setCopStabilityConstraint(double margin,
													const SoftConstraintProperties& properties)
{
	support_margin_ = margin;
	polygon_constraint_.setSoftProperties(properties);
}


void PreviewOptimization::setPreviewModelConstraint(double min_length,
													double max_length,
													double max_pitch,
													double max_roll,
													const SoftConstraintProperties& properties)
{
	// Getting the bounds
	Eigen::VectorXd lower_bound = Eigen::Vector3d(min_length, 0., 0.);
	Eigen::VectorXd upper_bound = Eigen::Vector3d(max_length,
												  max_pitch,
												  max_roll);

	// Computing the phase cost
	preview_constraint_ = ocp::PointConstraint(lower_bound, upper_bound);
	preview_constraint_.setSoftProperties(properties);
}


void PreviewOptimization::setVelocityWeights(double velocity)
{
	command_weight_ = Eigen::Vector2d(velocity, velocity);
}


void PreviewOptimization::setCostOfTransportWeight(double weight)
{
	cot_weight_ = weight;
}


void PreviewOptimization::setTerrainModel(const TerrainModel& model)
{
	terrain_model_ = model;
}


void PreviewOptimization::setVelocityCommand(double velocity_x,
											 double velocity_y)
{
	actual_command_ =
			simulation::VelocityCommand(Eigen::Vector2d(velocity_x, velocity_y), 0.);
}


void PreviewOptimization::getStartingPoint(double* decision, int decision_dim)
{
	// Eigen interfacing to raw buffers
	Eigen::Map<Eigen::VectorXd> full_initial_point(decision, decision_dim);

	// Getting the dimension of decision variables
	unsigned int control_dim = getDimensionOfState();
	full_initial_point = Eigen::VectorXd::Zero(control_dim);

	// Setting up the starting point
	// Setting up the starting values per every phase
	simulation::PreviewControl initial_control;
	initial_control.params.resize(phases_);
	for (unsigned int i = 0; i < phases_; ++i) {
		// Setting up the initial preview params values
		if (!warm_point_) {
			initial_control.params[i] =
					simulation::PreviewParams(0.1, Eigen::Vector2d::Zero());
		} else {
			initial_control.params[i] = warm_control_.params[i];
		}

		// Setting up the phase information
		initial_control.params[i].phase = schedule_.getPhase(i);

		unsigned int num_swings = schedule_.getNumberOfSwingFeet(i);
		for (unsigned int f = 0; f < num_swings; ++f) {
			std::string name = schedule_.getSwingFeet(i)[f];
			initial_control.params[i].phase.setFootShift(name, Eigen::Vector2d::Zero());
		}
	}

	// Converting the preview control to a vector
	Eigen::VectorXd control_vec;
	fromPreviewControl(control_vec, initial_control);

	// Appending the actual bounds
	full_initial_point = control_vec;
}


simulation::PreviewControl& PreviewOptimization::getFullPreviewControl()
{
	return full_pc_;
}


simulation::PreviewControl& PreviewOptimization::getAppliedPreviewControl()
{
	return applied_pc_;
}


simulation::VelocityCommand& PreviewOptimization::getVelocityCommand()
{
	return actual_command_;
}


void PreviewOptimization::evaluateBounds(double* decision_lbound, int decision_dim1,
										 double* decision_ubound, int decision_dim2,
										 double* constraint_lbound, int constraint_dim1,
										 double* constraint_ubound, int constraint_dim2)
{
	// Eigen interfacing to raw buffers
	Eigen::Map<Eigen::VectorXd> full_state_lower_bound(decision_lbound, decision_dim1);
	Eigen::Map<Eigen::VectorXd> full_state_upper_bound(decision_ubound, decision_dim2);
	Eigen::Map<Eigen::VectorXd> full_constraint_lower_bound(constraint_lbound, constraint_dim1);
	Eigen::Map<Eigen::VectorXd> full_constraint_upper_bound(constraint_ubound, constraint_dim2);

	// Defining the bound values using the PreviewBound structure
	// Setting up the preview bounds as a preview control bounds
	if (is_bound_) {
		simulation::PreviewControl low_control, upp_control;

		// Preview params bounds
		low_control.params.resize(phases_);
		upp_control.params.resize(phases_);
		for (unsigned int i = 0; i < phases_; ++i) {
			// Setting the bound of the preview params
			low_control.params[i] = simulation::PreviewParams(bounds_.lower.duration,
															  bounds_.lower.cop_shift);
			upp_control.params[i] = simulation::PreviewParams(bounds_.upper.duration,
															  bounds_.upper.cop_shift);

			// Setting up the phase information
			low_control.params[i].phase = schedule_.getPhase(i);
			upp_control.params[i].phase = schedule_.getPhase(i);

			// Foothold shift bounds
			unsigned int num_swings = schedule_.getNumberOfSwingFeet(i);
			for (unsigned int f = 0; f < num_swings; f++) {
				std::string name = schedule_.getSwingFeet(i)[f];
				low_control.params[i].phase.setFootShift(name, bounds_.lower.foothold_shift);
				upp_control.params[i].phase.setFootShift(name, bounds_.upper.foothold_shift);
			}
		}

		// Getting the bounds for the actual preview in the defined horizon
		Eigen::VectorXd actual_lower_bound, actual_upper_bound;
		fromPreviewControl(actual_lower_bound, low_control);
		fromPreviewControl(actual_upper_bound, upp_control);

		// Saving the bound vectors
		full_state_lower_bound = actual_lower_bound;
		full_state_upper_bound = actual_upper_bound;
	} else
		printf(YELLOW "Warning: it wasn't defined the boundaries\n" COLOR_RESET);
}


void PreviewOptimization::evaluateConstraints(double* constraint, int constraint_dim,
						 	 				  const double* decision, int decision_dim)
{

}


void PreviewOptimization::evaluateCosts(double& cost,
						   				const double* decision, int decision_dim)
{
	// Eigen interfacing to raw buffers
	const Eigen::Map<const Eigen::VectorXd> decision_var(decision, decision_dim);

	if (state_dimension_ != decision_var.size()) {
		printf(RED "FATAL: the control sequence and decision dimensions are"
				" not consistent\n" COLOR_RESET);
		exit(EXIT_FAILURE);
	}
	Eigen::VectorXd decision_state = decision_var;

	// Initializing the cost value
	cost = 0;

	// Computing the cost
	// Converting the decision state, for a certain time, to
	// preview control
	simulation::PreviewControl preview_control, nom_control;
	toPreviewControl(nom_control, decision_state);
	orderPreviewControl(preview_control, nom_control);

	// Getting the phase transitions for computing the cost
	// and soft-constraint functions
	ReducedBodyTrajectory phase_trans;
	preview_.multiPhasePreview(phase_trans,
							   actual_state_,
							   preview_control,
							   false);

	// Computing the velocity command cost
	ReducedBodyState terminal_state = phase_trans.back();
	double cmd_cost = velocityCost(terminal_state,
								   actual_state_,
								   preview_control);

	// Computing the CoM energy cost
	double energy_cost = costOfTransport(phase_trans.back(), actual_state_);

	// Computing the CoP stability constraint
	double cop_cost = copStabilitySoftConstraint(phase_trans,
												 actual_state_,
												 preview_control);

	// Computing the preview model constraint
	double prev_cost = previewModelSoftConstraint(phase_trans,
												  preview_control);

	// Computing the terrain cost
	double terrain_cost = terrainCost(phase_trans,
			  	  	  	  	  	  	  preview_control);


	// Summing up all the costs
	cost = cmd_cost + energy_cost + cop_cost + prev_cost + terrain_cost;

	if (collect_data_) {
		// Adding the information to be collected
		utils::CollectData::Dict data;
		data["cmd_cost"] = cmd_cost;
		data["energy_cost"] = energy_cost;
		data["cop_cost"] = cop_cost;
		data["prev_cost"] = prev_cost;
		data["terrain_cost"] = terrain_cost;
		data["total_cost"] = cost;
		for (unsigned int n = 0; n < getControlDimension(); n++)
			data["x_" +  std::to_string(n)] = decision_state(n);
		cdata_.writeNewData(data);
	}
}


WholeBodyTrajectory& PreviewOptimization::evaluateSolution(const Eigen::Ref<const Eigen::VectorXd>& solution)
{
	// Converting the solution state, for a certain time, to
	// preview control
	simulation::PreviewControl nom_control;
	Eigen::VectorXd solution_state = solution;
	toPreviewControl(nom_control, solution_state);

	// Getting the preview control sequence according the actual phase
	orderPreviewControl(full_pc_, nom_control);

	// Computing the applied control params
	unsigned int next_idx;
	if (num_controls_ != 0) {
		applied_pc_.params.resize(num_controls_);
		for (unsigned int u = 0; u < num_controls_; ++u)
			applied_pc_.params[u] = full_pc_.params[u];

		next_idx = (num_controls_ + phases_) % phases_;
	} else {
		applied_pc_ = full_pc_;
		next_idx = 0;
	}

	// Generating the multi-phase preview
	preview_.multiPhasePreview(reduced_trajectory_,
							   actual_state_,
							   applied_pc_);

	// Getting the preview transitions
	phase_transitions_.clear();
	preview_.multiPhasePreview(phase_transitions_,
							   actual_state_,
							   applied_pc_,
							   false);

	// Removing the swing feet of the next preview control action
	simulation::PreviewPhase next_phase = full_pc_.params[next_idx].phase;
	for (unsigned int f = 0; f < num_feet_; ++f) {
		std::string foot = feet_[f];

		if (next_phase.isSwingFoot(foot)) {
			reduced_trajectory_.back().support_region.erase(foot);
			phase_transitions_.back().support_region.erase(foot);
		}
	}


	// Converting the preview trajectory to whole-body trajectory
	preview_.toWholeBodyTrajectory(motion_solution_, reduced_trajectory_);

	// Setting up the last base velocities and accelerations states to
	// zero. This is a sanity measurement in cases that there is not an
	// update from the planner
	motion_solution_.back().setBaseVelocity_W(dwl::Motion());
	motion_solution_.back().setBaseAcceleration_W(dwl::Motion());

	return motion_solution_;
}


simulation::PreviewLocomotion* PreviewOptimization::getPreviewSystem()
{
	return &preview_;
}


WholeBodyTrajectory& PreviewOptimization::getWholeBodyTrajectory()
{
	return motion_solution_;
}


ReducedBodyTrajectory& PreviewOptimization::getReducedBodyTrajectory()
{
	return reduced_trajectory_;
}


ReducedBodyTrajectory& PreviewOptimization::getReducedBodySequence()
{
	reduced_sequence_.resize(phase_transitions_.size() + 1);

	// Adding the actual state
	reduced_sequence_[0] = actual_state_;

	for (unsigned int k = 0; k < phase_transitions_.size(); ++k)
		reduced_sequence_[k+1] = phase_transitions_[k];

	return reduced_sequence_;
}


void PreviewOptimization::saveControl(const simulation::PreviewData& data,
				 	 	 	 	 	  std::string filename)
{
	// Frame transformer
	math::FrameTF tf;

	// Recording the optimized preview sequence
	YAML::Emitter out;
	out << YAML::BeginMap;
	out << YAML::Comment("This file contains per every datapoint:\n"
						 "  (a) desired human commands (command)\n"
						 "  (b) the initial state conditions (state)\n"
						 "  (c) the preview sequence of parameters (preview_control)\n"
						 "Note that:\n"
						 "	- linear is desired 2d linear velocity (m/s).\n"
						 "	- angular is the desired yaw rate (rad/s) which is expressed\n"
						 "	  in the horizontal frame.\n"
						 "	- height represents the pendulum height used for generating\n"
						 "	  the following preview control sequence.\n"
						 "	- com_pos is the position relative to the CoP, and expressed\n"
						 "	  in the horizontal frame.\n"
						 "	- com_vel is expressed in the horizontal frame.\n"
						 "	- support describes the active feet.\n"
						 "	- number_phase describes how many the number of optimized\n"
						 "	  control actions.\n"
						 "	- duration describes phase duration.\n"
						 "	- cop_shift describes the displacement of the CoP in that\n"
						 "	  phase, which is expressed in the horizontal frame.\n"
						 "	- $(name_foot) describes the footshift applied to that\n"
						 "	  specific phase. The footshift is expressed in the stance\n"
						 "	  frame.\n");
	out << YAML::Key << "preview_sequence";
	out << YAML::Value << YAML::BeginMap;

	// Recording the number of data points
	out << YAML::Key << "number_datapoint";
	out << YAML::Value << data.size();

	for (unsigned int p = 0; p < data.size(); p++) {
		// Recording the actual data point
		out << YAML::Key << "datapoint_" + std::to_string(p);
		out << YAML::Value << YAML::BeginMap;

		// Recording the human command (input)
		out << YAML::Key << "command";
		out << YAML::Value << YAML::BeginMap;
		out << YAML::Key << "linear";
		out << YAML::Value << data[p].command.linear;
		out << YAML::Key << "angular";
		out << YAML::Value << data[p].command.angular;
		out << YAML::EndMap; // command ns

		// Recording the initial state (input)
		out << YAML::Key << "state";
		out << YAML::Value << YAML::BeginMap;
		simulation::PreviewState state = data[p].state;
		out << YAML::Key << "height";
		out << YAML::Value << state.height;
		out << YAML::Key << "com_pos";
		out << YAML::Value << state.com_pos;
		out << YAML::Key << "com_vel";
		out << YAML::Value << state.com_vel;
		out << YAML::Key << "support";
		out << YAML::Value << YAML::Flow << YAML::BeginSeq;
		for (simulation::SupportIterator it = state.support.begin();
						it != state.support.end(); it++) {
			if (it->second)
				out << it->first;
		}
		out << YAML::EndSeq; // support ns
		out << YAML::EndMap; // state ns

		// Recording the preview control (output)
		out << YAML::Key << "preview_control";
		out << YAML::Value << YAML::BeginMap;
		out << YAML::Key << "number_phase";
		simulation::PreviewControl control = data[p].control;
		out << YAML::Value << control.params.size();
		for (unsigned int k = 0; k < control.params.size(); ++k) {
			// Getting the actual preview params
			simulation::PreviewParams params = control.params[k];

			// Recording the actual preview params
			out << YAML::Key << "phase_" + std::to_string(k);
			out << YAML::Value << YAML::BeginMap;

			out << YAML::Key << "duration";
			out << YAML::Value << params.duration;
			if (params.phase.type == simulation::STANCE) {
				out << YAML::Key << "cop_shift";
				out << YAML::Value << params.cop_shift;

				for (unsigned int f = 0; f < params.phase.feet.size(); f++) {
					std::string name = params.phase.feet[f];
					Eigen::Vector2d foot_shift = params.phase.getFootShift(name);

					out << YAML::Key << name;
					out << YAML::Value << foot_shift;
				}
			}
			out << YAML::EndMap; // phase_k ns
		}
		out << YAML::EndMap; // preview_control ns
		out << YAML::EndMap; // point_p ns
	}

	std::cout << "Here's the output YAML:\n" << out.c_str() << std::endl;

	std::ofstream fout;
	fout.open(filename.c_str(), std::fstream::out);
	fout << out.c_str();
	fout.close();
}


void PreviewOptimization::saveSolution(std::string filename)
{
	simulation::PreviewState actual_state(actual_state_);
	simulation::PreviewSets point(actual_command_, actual_state, applied_pc_);
	simulation::PreviewData data(1, point);
	saveControl(data, filename);
}


double PreviewOptimization::velocityCost(const ReducedBodyState& terminal_state,
										 const ReducedBodyState& actual_state,
										 const simulation::PreviewControl& preview_control)
{
	double cost = 0.;
	// Computing the average velocity
	double travel_duration = preview_control.getTotalDuration();
	Eigen::Vector2d travel_distance =
			(terminal_state.getCoMSE3().getTranslation() -
					actual_state.getCoMSE3().getTranslation()).head<2>();
	Eigen::Vector2d avg_vel = travel_distance / travel_duration;

	// Computing the command cost
	Eigen::Vector2d error_vel = actual_command_.linear - avg_vel;
	cost = command_weight_.transpose() * error_vel.cwiseProduct(error_vel);

	return cost;
}


double PreviewOptimization::costOfTransport(const ReducedBodyState& terminal_state,
											const ReducedBodyState& actual_state)
{
	double travel_distance =
			(terminal_state.getCoMSE3().getTranslation() -
					actual_state.getCoMSE3().getTranslation()).head<2>().norm();

	// CoT = Kinetic Energy / (mass * gravity * travel distance)
	Eigen::Vector3d phase_vel_H =
			terminal_state.getCoMVelocity_H().getLinear() -
			actual_state.getCoMVelocity_H().getLinear();
	double CoT = cot_weight_ *
			phase_vel_H.squaredNorm() / (9.81 * travel_distance);

	return CoT;
}


double PreviewOptimization::copStabilitySoftConstraint(const ReducedBodyTrajectory& phase_trans,
													   const ReducedBodyState& actual_state,
													   const simulation::PreviewControl& preview_control)
{
	double cost = 0.;

	// Computing the cost associated to the stability
	Eigen::Vector3d previous_cop = actual_state.cop;
	for (unsigned int k = 0; k < preview_control.params.size(); ++k) {
		// Getting the actual preview phase and end state
		simulation::PreviewPhase phase = preview_control.params[k].phase;
		ReducedBodyState end_state = phase_trans[k];

		// Computing the CoP stability soft-constraint
		if (phase.type == simulation::STANCE) {
			// Getting the support polygon
			std::vector<Eigen::Vector3d> support;
			for (dwl::SE3Map::const_iterator
					vertex_it = end_state.support_region.begin();
					vertex_it != end_state.support_region.end(); ++vertex_it) {
				support.push_back(vertex_it->second.getTranslation());
			}

			// Getting the terminal CoP positions
			Eigen::Vector3d cop = end_state.getCoPPosition_W();

			// Computing the soft-constraint associated to the inequality
			// constraints products of imposing the CoP position inside the
			// support polygon
			// Terminal CoP position
			double phase_cost = 0.;
			ocp::PolygonState state(cop, support, support_margin_);
			polygon_constraint_.computeSoft(phase_cost, state);

			// Summing up the phase cost
			cost += phase_cost;

			// Previous CoP position //TODO Consider the flight phases
			if (k != 0) { // No for the initial phase. The actual CoP position
						  // has to be inside the support polygon
				state.point = previous_cop;
				polygon_constraint_.computeSoft(phase_cost, state);

				// Summing up the phase cost
				cost += phase_cost;
			}

			// Updating the previous CoP position
			previous_cop = cop;
		}
	}

	return cost;
}


double PreviewOptimization::previewModelSoftConstraint(const ReducedBodyTrajectory& phase_trans,
													   const simulation::PreviewControl& preview_control)
{
	double cost = 0.;
	for (unsigned int k = 0; k < preview_control.params.size(); k++) {
		// Getting the actual preview phase and end state
		simulation::PreviewPhase phase = preview_control.params[k].phase;
		ReducedBodyState end_state = phase_trans[k];

		// Computing the preview model soft-constraint
		if (phase.type == simulation::STANCE) {
			Eigen::Vector3d pendulum =
					end_state.getCoMSE3().getTranslation() -
					end_state.getCoPPosition_W();

			// Computing the target pendulum length and angles in stance phases
			double pendulum_length = pendulum.norm();
			double pitch_angle = atan(fabs((double) (pendulum(rbd::X) / pendulum(rbd::Z))));
			double roll_angle = atan(fabs((double) (pendulum(rbd::Y) / pendulum(rbd::Z))));
			Eigen::VectorXd state_model = Eigen::Vector3d(pendulum_length,
														  pitch_angle,
														  roll_angle);

			// Computing the phase cost
			double phase_cost = 0.;
			preview_constraint_.computeSoft(phase_cost, state_model);

			cost += phase_cost;
		}
	}

	return cost;
}


double PreviewOptimization::terrainCost(const ReducedBodyTrajectory& phase_trans,
										const simulation::PreviewControl& preview_control)
{
	double cost = 0.;

	// Reading the terrain cost if it's exist
	if (preview_.getTerrainMap()->isTerrainInformation()) {
		for (unsigned int k = 0; k < preview_control.params.size(); k++) {
			// Getting the actual preview phase and end state
			simulation::PreviewPhase phase = preview_control.params[k].phase;
			ReducedBodyState end_state = phase_trans[k+1];

			//
			if (phase.type == simulation::STANCE) {
				for (dwl::SE3Map::const_iterator it = end_state.support_region.begin();
						it != end_state.support_region.end(); ++it) {
					std::string name = it->first;

					// Adding the cost only for the swing feet
					if (phase.isSwingFoot(name)) {
						// Getting the foothold positions
						Eigen::Vector2d foothold_2d =
								it->second.getTranslation().head<2>();

						// Adding the terrain cost of the this foothold
						double terrain_cost =
								preview_.getTerrainMap()->getTerrainCost(foothold_2d);
						cost += terrain_model_.weight * terrain_cost;
						if (terrain_cost > terrain_model_.margin) // soft-constraint
							cost += terrain_model_.offset;
					}
				}
			}
		}
	}

	return cost;
}


unsigned int PreviewOptimization::getControlDimension()
{
	if (!set_schedule_) {
		printf(RED "Error: there is not defined the preview schedule \n"
				COLOR_RESET);
		return 0;
	}

	unsigned int control_dim = 0;
	for (unsigned int k = 0; k < phases_; k++)
		control_dim += getParamsDimension(k);

	return control_dim;
}


unsigned int PreviewOptimization::getParamsDimension(const unsigned int& phase)
{
	if (!set_schedule_) {
		printf(RED "Error: there is not defined the preview schedule\n"
				COLOR_RESET);
		return 0;
	}

	unsigned int phase_dim = 0;
	switch (schedule_.getTypeOfPhase(phase)) {
		case simulation::STANCE:
			// 4 decision variables per stance [duration, cop_shift]
			// plus 2 decision variables per feet
			phase_dim = 3 + 2 * schedule_.getNumberOfSwingFeet(phase);
			break;
		case simulation::FLIGHT:
			// Just the stance duration is a decision variable
			phase_dim = 1;
			break;
		default:
			printf(YELLOW "Warning: could not find the type of preview model\n");
			break;
	}

	return phase_dim;
}


void PreviewOptimization::orderPreviewControl(simulation::PreviewControl& control,
											  const simulation::PreviewControl& nom_control)
{
	// Converting the preview params for every phase
	control.params.resize(phases_);
	for (unsigned int k = 0; k < phases_; k++) {
		// Getting the phase id
		unsigned int id = phase_id_[k];

		// Filling the control params
		control.params[k] = nom_control.params[id];
	}
}


void PreviewOptimization::toPreviewControl(simulation::PreviewControl& preview_control,
										   const Eigen::VectorXd& generalized_control)
{

	if (getControlDimension() != generalized_control.size()) {
		printf(RED "FATAL: the preview-control and decision dimensions are not"
				" consistent\n" COLOR_RESET);
		exit(EXIT_FAILURE);
	}

	// Resizing the number of preview control phases
	preview_control.params.resize(phases_);

	// Converting the preview params for every phase
	unsigned int actual_idx = 0;
	for (unsigned int k = 0; k < phases_; ++k) {
		simulation::PreviewParams params;

		// Getting the preview params dimension for the actual phase
		unsigned int params_dim = getParamsDimension(k);

		// Converting the decision control vector, for a certain time,
		// to decision params
		Eigen::VectorXd decision_params =
				generalized_control.segment(actual_idx, params_dim);

		// Setting up the phase information
		params.id = k;
		params.phase = schedule_.getPhase(k);

		// Converting the generalized param vector to preview params
		if (params.phase.type == simulation::STANCE) {
			params.duration = decision_params(0);
			params.cop_shift = decision_params.segment<2>(1);

			// Adding the foothold target positions to the preview params
			for (unsigned int f = 0; f < params.phase.feet.size(); ++f) {
				std::string foot_name = params.phase.feet[f];
				Eigen::Vector2d foot_shift = decision_params.segment<2>(2 * f + 3);
				params.phase.setFootShift(foot_name, foot_shift);
			}
		} else {// Flight phase
			params.duration = decision_params(0);
			params.cop_shift = Eigen::Vector2d::Zero();
		}

		// Adding the actual preview params to the preview control vector
		preview_control.params[k] = params;

		// Updating the actual index of the decision vector
		actual_idx += params_dim;
	}
}


void PreviewOptimization::fromPreviewControl(Eigen::VectorXd& generalized_control,
											 const simulation::PreviewControl& preview_control)
{
	// Resizing the generalized control vector
	generalized_control.resize(getControlDimension());

	// Converting the preview params for every phase
	unsigned int actual_idx = 0;
	for (unsigned int k = 0; k < phases_; ++k) {
		// Getting the phase parameters
		simulation::PreviewParams params = preview_control.params[k];

		// Appending the preview duration
		generalized_control(actual_idx) = params.duration;
		actual_idx += 1;

		// Appending the preview parameters for the stance phase
		if (params.phase.type == simulation::STANCE) {
			generalized_control.segment<2>(actual_idx) = params.cop_shift;
			actual_idx += 2;

			// Appending the preview foothold shift to the preview params
			for (unsigned int f = 0; f < params.phase.feet.size(); ++f) {
				std::string foot_name = params.phase.feet[f];
				Eigen::Vector2d foot_shift = params.phase.getFootShift(foot_name);

				generalized_control.segment<2>(actual_idx) = foot_shift;
				actual_idx += 2;
			}
		}
	}
}

} //@namespace ocp
} //@namespace dwl
