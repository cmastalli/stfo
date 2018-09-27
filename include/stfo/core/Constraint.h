#ifndef DWL__OCP__CONSTRAINT__H
#define DWL__OCP__CONSTRAINT__H

#include <dwl/utils/utils.h>
#include <boost/shared_ptr.hpp>
#include <boost/circular_buffer.hpp>

#define NO_BOUND 2e19


namespace dwl
{

namespace ocp
{

enum SoftConstraintFamily {QUADRATIC, UNWEIGHTED};

struct SoftConstraintProperties
{
	SoftConstraintProperties() : weight(0.), threshold(0.), offset(0.),
			family(QUADRATIC) {}
	SoftConstraintProperties(double _weight,
							 double _threshold,
							 double _offset,
							 enum SoftConstraintFamily _family = QUADRATIC) :
								 weight(_weight),
								 threshold(_threshold),
								 offset(_offset),
								 family(_family) {}

	double weight;
	double threshold;
	double offset;
	enum SoftConstraintFamily family;
};

/**
 * @class Constraint
 * @brief Abstract class for defining constraints used in an
 * optimization-based locomotion approach
 */
template <class TState>
class Constraint
{
	public:
		/** @brief Constructor function */
		Constraint();

		/** @brief Destructor function */
		virtual ~Constraint();

		/** @brief Sets the constraint as soft constraint, i.e. inside the
		 * cost function */
		void defineAsSoftConstraint();

		/** @brief Sets the constraint as hard constraint, i.e. inside the
		 * cost function */
		void defineAsHardConstraint();

		/**
		 * @brief Initializes the constraint properties given an URDF model (xml)
		 * @param Print model information
		 */
		virtual void init(bool info = false);

		/**
		 * @brief Computes the soft-value of the constraint given a certain state
		 * @param double& Soft-value or the associated cost to the constraint
		 * @param const TState& Whole-body state
		 */
		void computeSoft(double& constraint_cost,
						 const TState& state);

		/**
		 * @brief Computes the constraint vector given a certain state
		 * @param Eigen::VectorXd& Evaluated constraint function
		 * @param const TState& Whole-body state
		 */
		virtual void compute(Eigen::VectorXd& constraint,
							 const TState& state) = 0;

		/**
		 * @brief Gets the lower and upper bounds of the constraint
		 * @param Eigen::VectorXd& Lower constraint bound
		 * @param Eigen::VectorXd& Upper constraint bound
		 */
		virtual void getBounds(Eigen::VectorXd& lower_bound,
							   Eigen::VectorXd& upper_bound) = 0;

		/** @brief Indicates is the constraint is implemented as soft-constraint */
		bool isSoftConstraint();

		/**
		 * @brief Sets the weight for computing the soft-constraint, i.e.
		 * the associated cost
		 * @param const SoftConstraintProperties& Soft-constraints properties
		 */
		void setSoftProperties(const SoftConstraintProperties& properties);

		/**
		 * @brief Sets the last state that could be used for the constraint
		 * @param TState& Last whole-body state
		 */
		void setLastState(TState& last_state);

		/** @brief Resets the state buffer */
		void resetStateBuffer();

		/** @brief Gets the dimension of the constraint */
		unsigned int getConstraintDimension();

		/**
		 * @brief Gets the name of the constraint
		 * @return The name of the constraint
		 */
		std::string getName();


	protected:
		/** @brief Name of the constraint */
		std::string name_;

		/** @brief Dimension of the constraint */
		unsigned int constraint_dimension_;

		/** @brief Label that indicates if it's implemented as soft constraint */
		bool is_soft_;

		/** @brief Soft-constraints properties */
		SoftConstraintProperties soft_properties_;

		/** @brief Sets the last state */
		boost::circular_buffer<TState> state_buffer_;
};

} //@namespace ocp
} //@namespace dwl

#include <dwl/ocp/impl/Constraint.hpp>

#endif
